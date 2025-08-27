from typing import List
from langchain_core.documents import Document

def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Helper function to format the retrieved documents,
    prepending a source identifier to each one.
    """
    return "\n\n---\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )