"""
Memory service handling conversation summarization and message pruning.
Implements strategic context management for long conversations.
"""

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    RemoveMessage,
)
from src.models.domain import AgentState
from src.services.llm_service import LLMService
from src.utils.prompts import load_prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROMPTS = load_prompts()


class MemoryService:
    """
    Service for conversation memory management including summarization
    and message pruning.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initialize memory service.

        Args:
            llm_service: LLM service for generating summaries
        """
        self.llm_service = llm_service

    async def summarize_conversation(self, state: AgentState) -> dict:
        """
        Creates or updates conversation summary and resets message window (async).

        Args:
            state: Current agent state

        Returns:
            Dictionary with updated summary, cleared messages, and reset turn count
        """
        logger.info("summarization_started", action="resetting_message_window")

        conversation_messages = state["messages"]
        current_summary = state.get("summary", "No hay resumen previo.")

        messages_text = "\n".join(
            f"- {type(msg).__name__}: {msg.content}"
            for msg in conversation_messages
        )

        prompt_template = PROMPTS["summarization"]["prompt_template"]
        system_message_text = PROMPTS["summarization"]["system_message"]

        summary_prompt = [
            SystemMessage(content=system_message_text),
            HumanMessage(
                content=prompt_template.format(
                    current_summary=current_summary, new_messages=messages_text
                )
            ),
        ]

        response = await self.llm_service.invoke_with_retry(summary_prompt)
        new_summary = response.content

        messages_to_remove = [
            RemoveMessage(id=msg.id) for msg in state["messages"]
        ]

        logger.info(
            "summarization_completed",
            messages_removed=len(messages_to_remove),
        )

        return {
            "summary": new_summary,
            "messages": messages_to_remove,
            "turn_count": 0,
        }

    def prune_tool_messages(self, state: AgentState) -> dict:
        """
        Removes intermediate tool-related messages after agent final response.
        Cleans AIMessage with tool_calls and corresponding ToolMessage.

        Args:
            state: Current agent state

        Returns:
            Dictionary with messages to remove, or empty dict if no pruning needed
        """
        logger.info("pruning_started", action="removing_tool_messages")

        messages = state["messages"]

        if not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
            return {}

        indices_to_remove = []
        for i in range(len(messages) - 2, -1, -1):
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                indices_to_remove.append(i)
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                indices_to_remove.append(i)
                break

        if not indices_to_remove:
            return {}

        message_ids_to_remove = [messages[i].id for i in indices_to_remove]
        messages_to_remove = [
            RemoveMessage(id=msg_id) for msg_id in message_ids_to_remove
        ]

        logger.info("pruning_completed", messages_pruned=len(messages_to_remove))
        return {"messages": messages_to_remove}

    @staticmethod
    def increment_turn_count(state: AgentState) -> dict:
        """
        Increments turn counter for conversation tracking.

        Args:
            state: Current agent state

        Returns:
            Dictionary with incremented turn_count
        """
        current_count = state.get("turn_count", 0)
        logger.info(
            "turn_count_incremented",
            previous_count=current_count,
            new_count=current_count + 1,
        )
        return {"turn_count": current_count + 1}
