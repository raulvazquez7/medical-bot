"""
Shared test fixtures and configuration.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document

from src.models.domain import AgentState


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    llm = Mock()
    llm.invoke = Mock(return_value=AIMessage(content="Mocked response"))
    return llm


@pytest.fixture
def mock_async_llm():
    """Mock async LLM for testing."""
    llm = AsyncMock()
    response = AIMessage(content="Mocked async response")
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    llm.invoke = AsyncMock(return_value=response)
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    embeddings = Mock()
    embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    embeddings.embed_documents = Mock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    return embeddings


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    client = Mock()
    
    # Mock RPC calls
    client.rpc = Mock()
    
    # Mock table operations
    table_mock = Mock()
    table_mock.delete = Mock(return_value=table_mock)
    table_mock.eq = Mock(return_value=table_mock)
    table_mock.insert = Mock(return_value=table_mock)
    table_mock.execute = Mock(return_value=Mock(data=[]))
    
    client.table = Mock(return_value=table_mock)
    
    return client


@pytest.fixture
def sample_agent_state() -> AgentState:
    """Sample AgentState for testing."""
    return {
        "messages": [
            HumanMessage(content="¿Qué es el ibuprofeno?"),
        ],
        "intent": None,
        "current_medicines": [],
        "summary": "",
        "turn_count": 0,
    }


@pytest.fixture
def sample_agent_state_with_history() -> AgentState:
    """AgentState with conversation history."""
    return {
        "messages": [
            HumanMessage(content="Hola"),
            AIMessage(content="¡Hola! ¿En qué puedo ayudarte?"),
            HumanMessage(content="¿Qué es el ibuprofeno?"),
            AIMessage(
                content="El ibuprofeno es un antiinflamatorio...",
                tool_calls=[
                    {
                        "name": "get_information_about_medicine",
                        "args": {"query": "¿Qué es el ibuprofeno?"},
                        "id": "call_1",
                    }
                ],
            ),
            ToolMessage(
                content="[Source 1]\nEl ibuprofeno es un AINE...",
                tool_call_id="call_1",
            ),
            AIMessage(content="El ibuprofeno es un antiinflamatorio..."),
        ],
        "intent": "pregunta_medicamento",
        "current_medicines": ["ibuprofeno"],
        "summary": "",
        "turn_count": 2,
    }


@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for RAG testing."""
    return [
        Document(
            page_content="El ibuprofeno es un antiinflamatorio no esteroideo (AINE).",
            metadata={
                "source": "ibuprofeno_cinfa_600.md",
                "medicine_name": "ibuprofeno",
                "section": "Qué es",
            },
        ),
        Document(
            page_content="Se utiliza para el alivio del dolor leve a moderado.",
            metadata={
                "source": "ibuprofeno_cinfa_600.md",
                "medicine_name": "ibuprofeno",
                "section": "Para qué se utiliza",
            },
        ),
    ]


@pytest.fixture
def known_medicines() -> list[str]:
    """List of known medicines for testing."""
    return [
        "ibuprofeno",
        "paracetamol",
        "nolotil",
        "lexatin",
        "orfidal",
    ]


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown content for chunking tests."""
    return """# 1. Qué es Ibuprofeno y para qué se utiliza

## Descripción

Ibuprofeno es un antiinflamatorio no esteroideo (AINE).

## Indicaciones

Se utiliza para:
- Alivio del dolor leve a moderado
- Reducción de la fiebre
- Tratamiento de la inflamación

# 2. Qué necesita saber antes de empezar a tomar Ibuprofeno

## Advertencias y precauciones

Consulte a su médico antes de tomar este medicamento si:
- Tiene problemas de estómago
- Tiene problemas cardíacos
"""


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Sample chunks for ingestion tests."""
    return [
        Document(
            page_content="Ibuprofeno es un antiinflamatorio no esteroideo (AINE).",
            metadata={
                "source": "ibuprofeno_600.md",
                "medicine_name": "ibuprofeno",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Se utiliza para el alivio del dolor leve a moderado.",
            metadata={
                "source": "ibuprofeno_600.md",
                "medicine_name": "ibuprofeno",
                "chunk_index": 1,
            },
        ),
    ]

