"""
Integration tests for LangGraph agent workflows.
Tests complete conversation flows through the graph.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.builder import build_graph
from src.models.domain import AgentState


@pytest.mark.integration
class TestGraphGreetingFlow:
    """Tests for greeting/farewell conversation flow."""

    @pytest.mark.asyncio
    @patch("src.graph.builder.create_client")
    @patch("src.graph.builder.get_embeddings_model")
    @patch("src.graph.builder.create_llm")
    async def test_greeting_flow_complete(
        self, mock_create_llm, mock_embeddings, mock_supabase
    ):
        """Should handle greeting through conversational node."""
        # Arrange - Mock dependencies
        mock_supabase.return_value = Mock()
        mock_supabase.return_value.rpc.return_value.execute.return_value = Mock(
            data=[{"medicine_name": "ibuprofeno"}]
        )

        mock_embeddings.return_value = Mock()

        # Mock router LLM (classifies intent)
        router_llm = Mock()
        router_llm.ainvoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "args": {
                        "intent": "saludo_despedida",
                        "medicine_name": None,
                    }
                }
            ],
        )

        # Mock agent LLM (not used in greeting flow)
        agent_llm = Mock()

        mock_create_llm.side_effect = [agent_llm, router_llm]

        # Build graph
        graph = build_graph(with_checkpointer=False)

        # Act
        inputs: AgentState = {
            "messages": [HumanMessage(content="Hola")],
        }

        result = await graph.ainvoke(inputs)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) >= 2  # HumanMessage + AIMessage
        assert isinstance(result["messages"][-1], AIMessage)
        # Should contain greeting response
        assert "hola" in result["messages"][-1].content.lower()


@pytest.mark.integration
class TestGraphMedicineQuestionFlow:
    """Tests for medicine question flow (happy path)."""

    @pytest.mark.asyncio
    @patch("src.graph.builder.create_client")
    @patch("src.graph.builder.get_embeddings_model")
    @patch("src.graph.builder.create_llm")
    async def test_known_medicine_retrieval_flow(
        self, mock_create_llm, mock_embeddings, mock_supabase
    ):
        """Should retrieve information for known medicine."""
        # Arrange
        # Mock Supabase
        supabase_mock = Mock()
        supabase_mock.rpc.return_value.execute.return_value = Mock(
            data=[
                {"medicine_name": "ibuprofeno"},
                {"medicine_name": "paracetamol"},
            ]
        )
        mock_supabase.return_value = supabase_mock
        
        # Mock embeddings
        embeddings_mock = Mock()
        embeddings_mock.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = embeddings_mock
        
        # Mock router LLM (classifies as medicine question)
        router_llm = Mock()
        router_llm.ainvoke.side_effect = [
            # First call: intent classification
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "args": {
                            "intent": "pregunta_medicamento",
                            "medicine_name": "ibuprofeno",
                        }
                    }
                ],
            ),
            # Second call: query rewriting
            AIMessage(content="¿Qué es el ibuprofeno y para qué se utiliza?"),
        ]

        # Mock agent LLM
        agent_llm = Mock()
        # First call: decides to use tool
        agent_llm.ainvoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_information_about_medicine",
                        "args": {"query": "¿Qué es el ibuprofeno?"},
                        "id": "call_123",
                    }
                ],
            ),
            # Second call: formulates final response
            AIMessage(
                content="El ibuprofeno es un antiinflamatorio no esteroideo (AINE)."
            ),
        ]

        agent_llm.bind_tools.return_value = agent_llm

        mock_create_llm.side_effect = [agent_llm, router_llm]

        # Mock retriever to return documents
        with patch("src.database.supabase.SupabaseRetriever._aget_relevant_documents") as mock_retriever:
            mock_retriever.return_value = [
                Mock(
                    page_content="El ibuprofeno es un AINE...",
                    metadata={"source": "ibuprofeno.md"},
                )
            ]

            # Build graph
            graph = build_graph(with_checkpointer=False)

            # Act
            inputs: AgentState = {
                "messages": [HumanMessage(content="¿Qué es el ibuprofeno?")],
            }

            result = await graph.ainvoke(inputs)
        
        # Assert
        assert "intent" in result
        assert result["intent"] == "pregunta_medicamento"
        assert "ibuprofeno" in result.get("current_medicines", [])
        
        # Should have final AI response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        assert len(ai_messages) > 0
        # Last AI message should not have tool_calls (final response)
        assert not ai_messages[-1].tool_calls


@pytest.mark.integration
class TestGraphUnauthorizedFlow:
    """Tests for unauthorized medicine questions."""

    @pytest.mark.asyncio
    @patch("src.graph.builder.create_client")
    @patch("src.graph.builder.get_embeddings_model")
    @patch("src.graph.builder.create_llm")
    async def test_unknown_medicine_blocked(
        self, mock_create_llm, mock_embeddings, mock_supabase
    ):
        """Should block questions about unknown medicines."""
        # Arrange
        mock_supabase.return_value = Mock()
        mock_supabase.return_value.rpc.return_value.execute.return_value = Mock(
            data=[{"medicine_name": "ibuprofeno"}]
        )

        mock_embeddings.return_value = Mock()

        # Mock router LLM
        router_llm = Mock()
        router_llm.ainvoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "args": {
                        "intent": "pregunta_medicamento",
                        "medicine_name": "aspirina",  # Not in known list
                    }
                }
            ],
        )

        agent_llm = Mock()
        mock_create_llm.side_effect = [agent_llm, router_llm]

        # Build graph
        graph = build_graph(with_checkpointer=False)

        # Act
        inputs: AgentState = {
            "messages": [HumanMessage(content="¿Qué es la aspirina?")],
        }

        result = await graph.ainvoke(inputs)

        # Assert
        assert result["intent"] == "pregunta_no_autorizada"
        # Should have unauthorized message
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert "no tengo información" in last_message.content.lower()


@pytest.mark.integration
class TestGraphMemoryFlow:
    """Tests for conversation memory and summarization."""

    def test_turn_count_increments(self):
        """Should increment turn count after each interaction."""
        # This test would require a more complex setup
        # with checkpointer to test state persistence
        # Skipping for now as it requires more infrastructure
        pytest.skip("Requires checkpointer setup")

