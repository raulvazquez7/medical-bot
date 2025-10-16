"""
Graph edge conditions for routing between nodes.
Contains all conditional logic for graph traversal.
"""

from langchain_core.messages import AIMessage, ToolMessage
from src.models.domain import AgentState
from src.utils.prompts import load_prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)


PROMPTS = load_prompts()
RETRIEVAL_FAILURE_MESSAGE = PROMPTS["constants"]["retrieval_failure_message"]
SUMMARIZATION_TURN_THRESHOLD = 3


def route_after_router(state: AgentState) -> str:
    """
    Routes based on classified intent after router node.
    Fails gracefully with default routing if state is invalid.

    Args:
        state: Current agent state

    Returns:
        Next node name: "conversational", "unauthorized", or "agent"
    """
    if not state.get("messages"):
        logger.error(
            "invalid_state_in_routing",
            error="State must contain messages",
            fallback="agent"
        )
        return "agent"

    intent = state.get("intent")
    if not intent:
        logger.warning("no_intent_in_state", fallback="agent")
        return "agent"

    if intent == "saludo_despedida":
        return "conversational"
    if intent == "pregunta_no_autorizada":
        return "unauthorized"
    return "agent"


def should_continue_react(state: AgentState) -> str:
    """
    Decides if agent should use tools or has finished reasoning.
    Fails gracefully if state is invalid.

    Args:
        state: Current agent state

    Returns:
        "tools" if agent wants to use tools, "end_of_turn" otherwise
    """
    if not state.get("messages"):
        logger.error(
            "invalid_state_in_react_check",
            error="State must contain messages",
            fallback="end_of_turn"
        )
        return "end_of_turn"

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end_of_turn"


def route_after_tools(state: AgentState) -> str:
    """
    Routes after tool execution based on retrieval success.
    Fails gracefully if state is invalid.

    Args:
        state: Current agent state

    Returns:
        "failure" if retrieval failed, "success" otherwise
    """
    if not state.get("messages"):
        logger.error(
            "invalid_state_in_tools_routing",
            error="State must contain messages",
            fallback="success"
        )
        return "success"

    last_message = state["messages"][-1]
    if (
        isinstance(last_message, ToolMessage)
        and last_message.content == RETRIEVAL_FAILURE_MESSAGE
    ):
        logger.info("retrieval_failure_detected", action="routing_to_failure_handler")
        return "failure"

    logger.info("retrieval_successful", action="returning_to_agent")
    return "success"


def should_summarize_or_end(state: AgentState) -> str:
    """
    Decides if conversation should be summarized based on turn count.

    Args:
        state: Current agent state

    Returns:
        "summarize" if threshold reached, "end" otherwise
    """
    turn_count = state.get("turn_count", 0)
    logger.info(
        "turn_count_check",
        turn_count=turn_count,
        threshold=SUMMARIZATION_TURN_THRESHOLD
    )

    if turn_count >= SUMMARIZATION_TURN_THRESHOLD:
        return "summarize"
    return "end"
