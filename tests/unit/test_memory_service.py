"""
Unit tests for MemoryService.
Tests conversation summarization, message pruning, and turn counting.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage,
)

from src.services.memory_service import MemoryService
from src.services.llm_service import LLMService
from src.models.domain import AgentState


@pytest.fixture
def llm_service():
    """Mock LLM service for testing."""
    return Mock(spec=LLMService)


@pytest.fixture
def memory_service(llm_service):
    """Create MemoryService with mocked dependencies."""
    return MemoryService(llm_service)


class TestConversationSummarization:
    """Tests for conversation summarization."""

    @pytest.mark.asyncio
    async def test_summarize_creates_new_summary(self, memory_service, llm_service):
        """Should create new summary from conversation."""
        # Arrange
        state: AgentState = {
            "messages": [
                HumanMessage(content="¿Qué es el ibuprofeno?", id="msg1"),
                AIMessage(content="Es un AINE...", id="msg2"),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 3,
        }

        llm_service.invoke_with_retry.return_value = AIMessage(
            content="Usuario preguntó sobre ibuprofeno, agente explicó que es un AINE."
        )

        # Act
        result = await memory_service.summarize_conversation(state)

        # Assert
        assert "summary" in result
        assert "ibuprofeno" in result["summary"].lower()
        assert result["turn_count"] == 0  # Reset counter
        assert len(result["messages"]) == 2  # 2 RemoveMessage instances
        assert all(isinstance(msg, RemoveMessage) for msg in result["messages"])

    @pytest.mark.asyncio
    async def test_summarize_updates_existing_summary(self, memory_service, llm_service):
        """Should update existing summary with new information."""
        # Arrange
        existing_summary = "Usuario preguntó sobre ibuprofeno."
        state: AgentState = {
            "messages": [
                HumanMessage(content="¿Tiene efectos secundarios?", id="msg3"),
                AIMessage(content="Sí, puede causar...", id="msg4"),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": existing_summary,
            "turn_count": 3,
        }

        new_summary = f"{existing_summary} Usuario preguntó sobre efectos secundarios."
        llm_service.invoke_with_retry.return_value = AIMessage(content=new_summary)

        # Act
        result = await memory_service.summarize_conversation(state)

        # Assert
        assert result["summary"] == new_summary
        assert "efectos secundarios" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_summarize_clears_all_messages(self, memory_service, llm_service):
        """Should remove all messages from state."""
        # Arrange
        state: AgentState = {
            "messages": [
                HumanMessage(content="Message 1", id="msg1"),
                AIMessage(content="Response 1", id="msg2"),
                HumanMessage(content="Message 2", id="msg3"),
                AIMessage(content="Response 2", id="msg4"),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 3,
        }

        llm_service.invoke_with_retry.return_value = AIMessage(content="Summary")

        # Act
        result = await memory_service.summarize_conversation(state)

        # Assert
        assert len(result["messages"]) == 4
        assert all(isinstance(msg, RemoveMessage) for msg in result["messages"])


class TestMessagePruning:
    """Tests for tool message pruning."""

    def test_prune_tool_messages_after_final_response(self, memory_service):
        """Should remove intermediate tool messages after agent response."""
        # Arrange
        state: AgentState = {
            "messages": [
                HumanMessage(content="¿Qué es el ibuprofeno?", id="msg1"),
                AIMessage(
                    content="",
                    id="msg2",
                    tool_calls=[
                        {
                            "name": "get_info",
                            "args": {"query": "ibuprofeno"},
                            "id": "call1",
                        }
                    ],
                ),
                ToolMessage(
                    content="[Source 1] Ibuprofeno es un AINE...",
                    tool_call_id="call1",
                    id="msg3",
                ),
                AIMessage(content="El ibuprofeno es un AINE...", id="msg4"),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 1,
        }
        
        # Act
        result = memory_service.prune_tool_messages(state)
        
        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 2  # Should remove AIMessage + ToolMessage
        assert all(isinstance(msg, RemoveMessage) for msg in result["messages"])

    def test_no_pruning_if_agent_still_has_tool_calls(self, memory_service):
        """Should not prune if last message has tool_calls."""
        # Arrange
        state: AgentState = {
            "messages": [
                HumanMessage(content="¿Qué es el ibuprofeno?", id="msg1"),
                AIMessage(
                    content="",
                    id="msg2",
                    tool_calls=[
                        {
                            "name": "get_info",
                            "args": {"query": "ibuprofeno"},
                            "id": "call1",
                        }
                    ],
                ),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 1,
        }
        
        # Act
        result = memory_service.prune_tool_messages(state)
        
        # Assert
        assert result == {}  # No pruning

    def test_no_pruning_if_no_tool_messages(self, memory_service):
        """Should not prune if there are no tool messages."""
        # Arrange
        state: AgentState = {
            "messages": [
                HumanMessage(content="Hola", id="msg1"),
                AIMessage(content="¡Hola!", id="msg2"),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 1,
        }
        
        # Act
        result = memory_service.prune_tool_messages(state)
        
        # Assert
        assert result == {}

    def test_prune_only_most_recent_tool_sequence(self, memory_service):
        """Should only prune the most recent tool call sequence."""
        # Arrange
        state: AgentState = {
            "messages": [
                # First tool sequence (should NOT be pruned)
                HumanMessage(content="Question 1", id="msg1"),
                AIMessage(content="", id="msg2", tool_calls=[{"id": "call1"}]),
                ToolMessage(content="Result 1", tool_call_id="call1", id="msg3"),
                AIMessage(content="Answer 1", id="msg4"),
                # Second tool sequence (should be pruned)
                HumanMessage(content="Question 2", id="msg5"),
                AIMessage(content="", id="msg6", tool_calls=[{"id": "call2"}]),
                ToolMessage(content="Result 2", tool_call_id="call2", id="msg7"),
                AIMessage(content="Answer 2", id="msg8"),
            ],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 2,
        }
        
        # Act
        result = memory_service.prune_tool_messages(state)
        
        # Assert
        assert len(result["messages"]) == 2  # Only msg6 and msg7


class TestTurnCounting:
    """Tests for turn counter management."""

    def test_increment_turn_count_from_zero(self, memory_service):
        """Should increment turn count from 0 to 1."""
        # Arrange
        state: AgentState = {
            "messages": [],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 0,
        }
        
        # Act
        result = memory_service.increment_turn_count(state)
        
        # Assert
        assert result["turn_count"] == 1

    def test_increment_turn_count_sequential(self, memory_service):
        """Should increment turn count sequentially."""
        # Arrange
        state: AgentState = {
            "messages": [],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 5,
        }
        
        # Act
        result = memory_service.increment_turn_count(state)
        
        # Assert
        assert result["turn_count"] == 6

    def test_increment_turn_count_handles_missing_key(self, memory_service):
        """Should handle missing turn_count key."""
        # Arrange
        state: AgentState = {
            "messages": [],
            "intent": None,
            "current_medicines": [],
            "summary": "",
        }
        
        # Act
        result = memory_service.increment_turn_count(state)
        
        # Assert
        assert result["turn_count"] == 1

