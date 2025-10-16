"""
Retrieval service handling RAG operations including search and query rewriting.
Coordinates between embeddings, database, and LLM for optimal retrieval.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage
from src.services.llm_service import LLMService
from src.utils.prompts import load_prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROMPTS = load_prompts()


class RetrievalService:
    """
    Service for RAG operations including document retrieval and query rewriting.
    """

    def __init__(self, retriever: BaseRetriever, llm_service: LLMService):
        """
        Initialize retrieval service.

        Args:
            retriever: Configured retriever instance (e.g., SupabaseRetriever)
            llm_service: LLM service for query rewriting
        """
        self.retriever = retriever
        self.llm_service = llm_service

    async def search_medicine_info(self, query: str) -> list[Document]:
        """
        Searches database for medicine information (async).

        Args:
            query: Search query

        Returns:
            List of relevant documents (empty if nothing found)

        Raises:
            Exception: Database errors are propagated (fail-fast)
        """
        logger.info("search_started", query=query)
        docs = await self.retriever.ainvoke(query)

        if not docs:
            logger.warning("search_completed", query=query, docs_found=0)
            return []

        logger.info("search_completed", query=query, docs_found=len(docs))
        return docs

    async def rewrite_query_with_context(
        self,
        original_query: str,
        conversation_history: list[BaseMessage],
        summary: str = "",
    ) -> str:
        """
        Rewrites query incorporating conversation context for better retrieval.
        Skips rewriting if no meaningful context exists (optimization).

        Args:
            original_query: Original user query
            conversation_history: Recent conversation messages
            summary: Conversation summary for long-term context

        Returns:
            Rewritten, context-enriched query (or original if no context)
        """
        has_context = bool(summary) or len(conversation_history) > 1
        if not has_context:
            logger.info(
                "query_rewrite_skipped",
                reason="no_context",
                original_query=original_query,
            )
            return original_query

        logger.info("query_rewrite_started", original_query=original_query)

        history_text = summary + "\n\n" if summary else ""
        history_text += "\n".join(
            f"{type(msg).__name__}: {msg.content}"
            for msg in conversation_history
        )

        prompt_template = PROMPTS["query_rewriter"]["prompt_template"]
        system_prompt = PROMPTS["query_rewriter"]["system_prompt"]

        full_prompt = (
            system_prompt
            + "\n\n"
            + prompt_template.format(
                conversation_history=history_text, query=original_query
            )
        )

        response = await self.llm_service.invoke_with_retry(full_prompt)
        rewritten_query = response.content.strip()

        logger.info(
            "query_rewrite_completed",
            original_query=original_query,
            rewritten_query=rewritten_query,
        )
        return rewritten_query


def format_docs_with_sources(docs: list[Document]) -> str:
    """
    Formats retrieved documents with source identifiers.

    Args:
        docs: List of documents to format

    Returns:
        Formatted string with numbered sources
    """
    if not docs:
        return PROMPTS["constants"]["retrieval_failure_message"]

    return "\n\n---\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )
