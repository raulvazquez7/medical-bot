"""
Graph builder for constructing the LangGraph agent workflow.
Assembles nodes, edges, and services into executable graph.
"""

from supabase import create_client
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src import config
from src.utils.logger import get_logger

logger = get_logger(__name__)
from src.models.domain import AgentState
from src.models.schemas import MedicineToolInput
from src.models.embeddings import get_embeddings_model
from src.database.supabase import SupabaseRetriever, get_known_medicines
from src.services.llm_service import LLMService, create_llm
from src.services.retrieval_service import (
    RetrievalService,
    format_docs_with_sources,
)
from src.services.medicine_service import MedicineService
from src.services.memory_service import MemoryService
from src.graph.nodes import GraphNodes
from src.graph.edges import (
    route_after_router,
    should_continue_react,
    route_after_tools,
    should_summarize_or_end,
)


def build_graph(with_checkpointer: bool = False):
    """
    Builds and compiles the LangGraph agent workflow.

    Args:
        with_checkpointer: If True, compiles with MemorySaver for persistence

    Returns:
        Compiled graph ready for execution
    """
    logger.info("graph_components_initializing")

    settings = config.get_settings()

    supabase = create_client(settings.supabase_url, settings.supabase_service_key)
    known_medicines = get_known_medicines(supabase)

    embeddings = get_embeddings_model(
        provider=config.EMBEDDINGS_PROVIDER.lower(),
        model=config.EMBEDDINGS_MODEL,
        api_key=(
            settings.google_api_key
            if config.EMBEDDINGS_PROVIDER.lower() == "google"
            else settings.openai_api_key
        ),
        cache_size=config.EMBEDDINGS_CACHE_SIZE,
    )

    supabase_retriever = SupabaseRetriever(
        supabase_client=supabase, embeddings_model=embeddings
    )

    agent_llm = create_llm(
        model_name=config.AGENT_MODEL,
        api_key=(
            settings.openai_api_key
            if "gpt" in config.AGENT_MODEL
            else settings.google_api_key
        ),
    )

    router_llm = create_llm(
        model_name=config.ROUTER_MODEL,
        api_key=(
            settings.openai_api_key
            if "gpt" in config.ROUTER_MODEL
            else settings.google_api_key
        ),
    )

    router_llm_service = LLMService(
        model=router_llm,
        max_retries=config.LLM_MAX_RETRIES,
        timeout=config.LLM_TIMEOUT,
        rate_limit=config.LLM_RATE_LIMIT,
    )

    retrieval_service = RetrievalService(supabase_retriever, router_llm_service)
    medicine_service = MedicineService(router_llm_service, known_medicines)
    memory_service = MemoryService(router_llm_service)

    async def medicine_tool_func(query: str) -> str:
        """Async tool function for medicine information retrieval."""
        docs = await retrieval_service.search_medicine_info(query)
        return format_docs_with_sources(docs)

    medicine_tool = Tool(
        name="get_information_about_medicine",
        description="Busca en la BBDD de prospectos información sobre un medicamento.",
        func=medicine_tool_func,
        args_schema=MedicineToolInput,
        coroutine=medicine_tool_func,
    )

    agent_llm_with_tools = agent_llm.bind_tools([medicine_tool])

    nodes = GraphNodes(
        medicine_service=medicine_service,
        retrieval_service=retrieval_service,
        memory_service=memory_service,
        agent_llm_with_tools=agent_llm_with_tools,
        rewriter_llm=router_llm,
    )

    tool_node = ToolNode([medicine_tool])

    logger.info("graph_workflow_building")
    workflow = StateGraph(AgentState)

    # Set recursion limit to prevent infinite ReAct loops
    workflow.recursion_limit = config.MAX_REACT_ITERATIONS

    workflow.add_node("router", nodes.router_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("query_rewriter", nodes.query_rewriter_node)
    workflow.add_node("conversational", nodes.conversational_node)
    workflow.add_node("unauthorized", nodes.unauthorized_question_node)
    workflow.add_node("handle_retrieval_failure", nodes.handle_retrieval_failure_node)
    workflow.add_node("summarizer", nodes.summarize_node)
    workflow.add_node("end_of_turn", nodes.end_of_turn_node)
    workflow.add_node("pruning", nodes.pruning_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "agent": "agent",
            "conversational": "conversational",
            "unauthorized": "unauthorized",
        },
    )

    workflow.add_conditional_edges(
        "agent",
        should_continue_react,
        {"tools": "query_rewriter", "end_of_turn": "pruning"},
    )

    workflow.add_conditional_edges(
        "end_of_turn",
        should_summarize_or_end,
        {"summarize": "summarizer", "end": END},
    )

    workflow.add_conditional_edges(
        "tools", route_after_tools, {"success": "agent", "failure": "handle_retrieval_failure"}
    )

    workflow.add_edge("query_rewriter", "tools")
    workflow.add_edge("conversational", "end_of_turn")
    workflow.add_edge("unauthorized", "end_of_turn")
    workflow.add_edge("handle_retrieval_failure", "end_of_turn")
    workflow.add_edge("pruning", "end_of_turn")
    workflow.add_edge("summarizer", END)

    if with_checkpointer:
        memory = MemorySaver()
        logger.info("graph_compiling", checkpointer=True)
        return workflow.compile(checkpointer=memory)
    else:
        logger.info("graph_compiling", checkpointer=False)
        return workflow.compile()
