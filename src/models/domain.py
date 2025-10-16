"""
Domain models representing the core business entities and state.
AgentState is the central state object for the LangGraph agent.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Represents the state of the medical chatbot agent.
    Used by LangGraph to maintain conversation context across nodes.

    Attributes:
        messages: Sliding window of recent conversation messages.
        summary: Long-term conversation summary for context management.
        turn_count: Explicit counter for triggering summarization.
        current_medicines: List of validated medicine names mentioned.
        intent: Classified intent of the last user message.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    turn_count: int
    current_medicines: list[str]
    intent: str
