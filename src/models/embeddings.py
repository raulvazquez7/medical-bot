"""
Embeddings model factory and custom wrappers.
Centralizes embeddings provider selection (OpenAI vs Google).
Includes LRU cache for query embeddings to reduce costs.
"""

from typing import Literal
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Wrapper around GoogleGenerativeAIEmbeddings to enforce consistent
    output dimensionality and appropriate task_type for each call.
    """

    output_dim: int = 1536

    def embed_query(self, text: str, **kwargs) -> list[float]:
        """Embeds a query with optimized parameters for search."""
        kwargs.pop("output_dimensionality", None)
        kwargs.pop("task_type", None)
        return super().embed_query(
            text=text,
            output_dimensionality=self.output_dim,
            task_type="retrieval_query",
            **kwargs,
        )

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Embeds documents with optimized parameters for storage."""
        kwargs.pop("output_dimensionality", None)
        kwargs.pop("task_type", None)
        return super().embed_documents(
            texts=texts,
            output_dimensionality=self.output_dim,
            task_type="retrieval_document",
            **kwargs,
        )


class CachedEmbeddingsWrapper(Embeddings):
    """
    Wrapper that adds LRU cache to embed_query for cost reduction.
    Documents are not cached as they're typically unique.
    """

    def __init__(self, base_embeddings: Embeddings, cache_size: int = 100):
        """
        Initialize cached embeddings wrapper.

        Args:
            base_embeddings: Underlying embeddings model
            cache_size: LRU cache size for query embeddings
        """
        self.base_embeddings = base_embeddings
        self._cached_embed_query = lru_cache(maxsize=cache_size)(
            self._embed_query_impl
        )

    def _embed_query_impl(self, text: str) -> tuple[float, ...]:
        """
        Internal implementation that gets cached.
        Returns tuple (hashable) instead of list for caching.
        """
        return tuple(self.base_embeddings.embed_query(text))

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds query with LRU caching.
        Identical queries return cached embeddings.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        return list(self._cached_embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds documents without caching (documents are typically unique).

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        return self.base_embeddings.embed_documents(texts)


def get_embeddings_model(
    provider: Literal["openai", "google"],
    model: str,
    api_key: str,
    cache_size: int = 100,
) -> Embeddings:
    """
    Factory function to create embeddings model with caching.

    Args:
        provider: Embeddings provider ("openai" or "google")
        model: Model identifier (e.g., "text-embedding-3-small")
        api_key: API key for the provider
        cache_size: LRU cache size for query embeddings (0 to disable)

    Returns:
        Configured embeddings model instance with caching

    Raises:
        ValueError: If provider is not supported
    """
    logger.info("embeddings_model_initializing", provider=provider, model=model)

    if provider == "google":
        base_embeddings = CustomGoogleEmbeddings(google_api_key=api_key, model=model)
    elif provider == "openai":
        base_embeddings = OpenAIEmbeddings(api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unsupported embeddings provider: {provider}. "
            "Use 'openai' or 'google'"
        )

    if cache_size > 0:
        logger.info("embeddings_cache_enabled", cache_size=cache_size)
        return CachedEmbeddingsWrapper(base_embeddings, cache_size)
    else:
        logger.info("embeddings_cache_disabled")
        return base_embeddings
