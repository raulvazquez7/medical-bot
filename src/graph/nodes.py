"""
Graph nodes implementing the agent's decision points.
Each node is thin and delegates business logic to services.
"""

from langchain_core.messages import AIMessage, SystemMessage
from src.models.domain import AgentState
from src.utils.prompts import load_prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROMPTS = load_prompts()


class GraphNodes:
    """
    Container for all graph node functions.
    Nodes are thin wrappers that delegate to services.
    """

    def __init__(
        self,
        medicine_service,
        retrieval_service,
        memory_service,
        agent_llm_with_tools,
        rewriter_llm,
    ):
        """
        Initialize graph nodes with required services.

        Args:
            medicine_service: Service for medicine operations
            retrieval_service: Service for RAG operations
            memory_service: Service for memory management
            agent_llm_with_tools: LLM bound with tools for agent
            rewriter_llm: LLM for query rewriting
        """
        self.medicine_service = medicine_service
        self.retrieval_service = retrieval_service
        self.memory_service = memory_service
        self.agent_llm_with_tools = agent_llm_with_tools
        self.rewriter_llm = rewriter_llm

    async def router_node(self, state: AgentState) -> dict:
        """
        Entry point node that classifies intent and validates medicine (async).
        Delegates to MedicineService.
        """
        logger.info("node_started", node="router", action="classifying_intent")
        return await self.medicine_service.classify_intent_and_validate(state)

    async def agent_node(self, state: AgentState) -> dict:
        """
        Main ReAct agent node that formulates responses using tools (async).
        """
        logger.info("node_started", node="agent", action="processing_with_tools")

        agent_system_prompt = PROMPTS["agent_system"]["content"]
        context = [SystemMessage(content=agent_system_prompt)]

        if state.get("summary"):
            context.append(
                SystemMessage(f"Resumen de la conversación:\n{state['summary']}")
            )

        context.extend(state["messages"])

        response = await self.agent_llm_with_tools.ainvoke(context)
        return {"messages": [response]}

    async def query_rewriter_node(self, state: AgentState) -> dict:
        """
        Rewrites agent's search query with conversation context (async).
        Skips rewriting if no context exists (first turn optimization).
        """
        logger.info("node_started", node="query_rewriter")

        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return {}

        original_tool_call = last_message.tool_calls[0]
        original_query = original_tool_call["args"]["query"]

        rewritten_query = await self.retrieval_service.rewrite_query_with_context(
            original_query=original_query,
            conversation_history=state["messages"][:-1],
            summary=state.get("summary", ""),
        )

        new_tool_calls = last_message.tool_calls.copy()
        new_tool_calls[0]["args"]["query"] = rewritten_query

        new_message = AIMessage(
            content=last_message.content,
            tool_calls=new_tool_calls,
            id=last_message.id,
        )

        all_but_last = state["messages"][:-1]
        return {"messages": all_but_last + [new_message]}

    def conversational_node(self, state: AgentState) -> dict:  # noqa: ARG002
        """Handles simple greetings and farewells."""
        logger.info("node_started", node="conversational", action="handling_greeting")
        greeting = PROMPTS["conversation_responses"]["greeting"]
        return {"messages": [AIMessage(content=greeting)]}

    def unauthorized_question_node(self, state: AgentState) -> dict:  # noqa: ARG002
        """Handles questions about unknown medications."""
        logger.info(
            "node_started", node="unauthorized", action="medicine_not_in_database"
        )
        content = self.medicine_service.get_unauthorized_medicine_message()
        return {"messages": [AIMessage(content=content)]}

    def handle_retrieval_failure_node(self, state: AgentState) -> dict:
        """Handles cases where database search returns no results."""
        logger.info("node_started", node="retrieval_failure", action="no_docs_found")

        last_known_medicine = (
            state.get("current_medicines", [])[-1]
            if state.get("current_medicines")
            else "el medicamento"
        )

        content = self.medicine_service.get_retrieval_failure_message(
            last_known_medicine
        )
        return {"messages": [AIMessage(content=content)]}

    async def summarize_node(self, state: AgentState) -> dict:
        """
        Summarizes conversation and resets message window (async).
        Delegates to MemoryService.
        """
        logger.info("node_started", node="summarize", action="creating_summary")
        return await self.memory_service.summarize_conversation(state)

    def end_of_turn_node(self, state: AgentState) -> dict:
        """Increments turn counter at end of conversation turn."""
        logger.info("node_started", node="end_of_turn", action="incrementing_counter")
        return self.memory_service.increment_turn_count(state)

    def pruning_node(self, state: AgentState) -> dict:
        """
        Prunes intermediate tool messages from history.
        Delegates to MemoryService.
        """
        logger.info("node_started", node="pruning", action="cleaning_tool_messages")
        return self.memory_service.prune_tool_messages(state)
