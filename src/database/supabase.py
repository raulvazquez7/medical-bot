"""
Supabase database integration with fail-fast error handling.
Provides retriever for RAG operations and medicine database queries.
"""

import asyncio
from supabase import Client
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Raised when database operations fail."""


class SupabaseRetriever(BaseRetriever):
    """
    Custom retriever that searches Supabase using vector similarity.
    Implements LangChain's BaseRetriever interface.
    """

    supabase_client: Client
    embeddings_model: Embeddings
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """
        Sync retrieval (fallback for compatibility).
        Prefer using async version via ainvoke().

        Args:
            query: User query to search for

        Returns:
            List of relevant documents with metadata

        Raises:
            DatabaseError: If embedding generation or search fails
        """
        try:
            logger.info("embedding_started_sync", query=query)
            query_embedding = self.embeddings_model.embed_query(query)

            rpc_params = {"query_embedding": query_embedding, "match_count": self.top_k}

            logger.info("database_search_started_sync", top_k=self.top_k)
            response = self.supabase_client.rpc("match_documents", rpc_params).execute()

            if response.data:
                logger.info("documents_found_sync", count=len(response.data))
                return [
                    Document(page_content=doc["content"], metadata=doc["metadata"])
                    for doc in response.data
                ]

            logger.warning("no_documents_found_sync")
            return []

        except Exception as e:
            logger.error("retrieval_failed_sync", exc_info=True, error=str(e))
            raise DatabaseError(f"Failed to retrieve documents: {e}") from e

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """
        Async retrieval implementation for true non-blocking operations.
        This is the preferred method for async contexts.

        Args:
            query: User query to search for

        Returns:
            List of relevant documents with metadata

        Raises:
            DatabaseError: If embedding generation or search fails
        """
        try:
            logger.info("embedding_started", query=query)
            query_embedding = await asyncio.to_thread(
                self.embeddings_model.embed_query, query
            )

            rpc_params = {"query_embedding": query_embedding, "match_count": self.top_k}

            logger.info("database_search_started", top_k=self.top_k)
            response = await asyncio.to_thread(
                lambda: self.supabase_client.rpc("match_documents", rpc_params).execute()
            )

            if response.data:
                logger.info("documents_found", count=len(response.data))
                return [
                    Document(page_content=doc["content"], metadata=doc["metadata"])
                    for doc in response.data
                ]

            logger.warning("no_documents_found")
            return []

        except Exception as e:
            logger.error("retrieval_failed", exc_info=True, error=str(e))
            raise DatabaseError(f"Failed to retrieve documents: {e}") from e


def get_known_medicines(client: Client) -> list[str]:
    """
    Fetches list of known medicine names from database.

    Args:
        client: Supabase client instance

    Returns:
        List of medicine names in lowercase

    Raises:
        DatabaseError: If database query fails
    """
    try:
        logger.info("fetching_known_medicines")
        response = client.rpc("get_distinct_medicine_names", {}).execute()

        if not response.data:
            logger.warning("no_medicines_found_in_database")
            return []

        medicines = [item["medicine_name"].lower() for item in response.data]
        logger.info("medicines_loaded", count=len(medicines))
        return medicines

    except Exception as e:
        logger.error("fetch_medicines_failed", exc_info=True, error=str(e))
        raise DatabaseError(f"Could not load medicine list: {e}") from e
