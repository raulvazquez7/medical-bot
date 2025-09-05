"""
This script defines and compiles the stateful, multi-actor agent using LangGraph.
It orchestrates the entire conversational flow, from intent classification to
tool use, memory management, and guardrails.
"""
import logging
import uuid
import functools
import re
from typing import TypedDict, Annotated, List, Literal, Optional

# LangChain & LangGraph specific imports
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool, Tool
from pydantic import BaseModel, Field


# Project specific imports
from src import config
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client as SupabaseClient
from src.database import SupabaseRetriever, get_known_medicines
from src.utils import format_docs_with_sources
from src.models import get_embeddings_model

# ==============================================================================
# === 1. AGENT STATE DEFINITION                                              ===
# ==============================================================================
class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        messages: A sliding window of the most recent messages.
        summary: The long-term summary of the conversation.
        turn_count: An explicit counter for conversation turns.
        current_medicines: A list of identified medications.
        intent: The classified intent of the last message.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    turn_count: int
    current_medicines: List[str]
    intent: str

# ==============================================================================
# === 2. TOOL AND INTENT DEFINITIONS                                         ===
# ==============================================================================

class MedicineToolInput(BaseModel):
    """Input schema for the medicine information retrieval tool."""
    query: str = Field(description="The specific question about the medicine to search for.")

# A constant to reliably identify a retrieval failure.
RETRIEVAL_FAILURE_MESSAGE = "No se encontró información relevante en la base de datos para esa consulta."

def get_information_about_medicine(query: str, retriever: SupabaseRetriever) -> str:
    """Searches the leaflet database for information about a medicine."""
    logging.info(f"--- Executing Search Tool for: '{query}' ---")
    docs = retriever.invoke(query)
    if not docs:
        return RETRIEVAL_FAILURE_MESSAGE
    return format_docs_with_sources(docs)

class UserIntent(BaseModel):
    """Classifies the user's intent to route the conversation."""
    intent: Literal[
        "pregunta_medicamento", 
        "pregunta_general", 
        "saludo_despedida"
    ] = Field(description="The main intent of the user's message.")
    
    medicine_name: Optional[str] = Field(
        description="The name of the medicine identified in the question, if any."
    )

# ==============================================================================
# === 3. MODEL CREATION HELPER                                               ===
# ==============================================================================

def create_llm(model_name: str) -> BaseChatModel:
    """
    Factory function to create a chat model instance based on the config name,
    centralizing the provider selection logic (OpenAI vs. Google).
    """
    if "gemini" in model_name:
        return ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY, model=model_name, temperature=0
        )
    elif "gpt" in model_name:
        return ChatOpenAI(
            api_key=config.OPENAI_API_KEY, model=model_name, temperature=0
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# ==============================================================================
# === 4. GRAPH NODE DEFINITIONS                                              ===
# ==============================================================================
SUMMARIZATION_TURN_THRESHOLD = 3

def router_node(state: AgentState, model: BaseChatModel, known_medicines: List[str]):
    """The entry point. Classifies intent and validates the medicine against a known list."""
    logging.info("--- Router: Classifying user intent ---")
    last_message = state['messages'][-1].content
    
    prompt = (
        f"Clasifica la pregunta de un usuario. Las categorías son: 'pregunta_medicamento', "
        f"'saludo_despedida', o 'pregunta_general' (para cualquier otra cosa). "
        f"Extrae el nombre del medicamento solo si la intención es 'pregunta_medicamento'.\n\n"
        f"Pregunta: '{last_message}'"
    )
    
    try:
        structured_response = model.invoke(prompt, tools=[UserIntent])
        if not structured_response.tool_calls:
            raise ValueError("Model did not return a structured intent.")
            
        intent_tool_call = structured_response.tool_calls[0]
        intent = intent_tool_call['args']['intent']
        medicine = intent_tool_call['args'].get('medicine_name', None)
        
        logging.info(f"--- LLM-classified intent: {intent} | Extracted medicine: {medicine} ---")
        
        current_medicines = state.get("current_medicines", [])
        
        # We check the extracted medicine against our trusted list.
        if medicine and intent == "pregunta_medicamento":
            medicine_lower = medicine.lower()
            
            # Stricter validation logic to avoid false positives (e.g., 'nolotil' vs 'nolotilazo').
            matched_medicine = None
            for known_med in known_medicines:
                if re.search(r'\b' + re.escape(known_med) + r'\b', medicine_lower):
                    matched_medicine = known_med
                    break 

            if matched_medicine:
                # If it's a known medicine, add it to the state if it's not already there.
                logging.info(f"--- Medicine '{matched_medicine}' validated. ---")
                if matched_medicine not in current_medicines:
                    current_medicines.append(matched_medicine)
            else:
                # If the LLM extracted something but we don't know it, change the intent.
                logging.warning(f"--- Medicine '{medicine_lower}' is not known. Rerouting to unauthorized_question. ---")
                intent = "pregunta_no_autorizada"
        
        return {"intent": intent, "current_medicines": current_medicines}

    except Exception as e:
        logging.error(f"Error in router: {e}. Defaulting to 'pregunta_general'.")
        return {"intent": "pregunta_general"}

# This system prompt acts as the agent's constitution, guiding its behavior.
AGENT_SYSTEM_PROMPT = """Eres un asistente médico virtual especializado en información de prospectos de medicamentos.

Tu objetivo principal es responder a las preguntas de los usuarios utilizando únicamente la información obtenida a través de la herramienta `get_information_about_medicine`.

Sigue estas reglas estrictamente:
1.  **Piensa paso a paso.** Antes de responder, analiza la pregunta del usuario.
2.  **Uso obligatorio de herramientas:** Si la pregunta es sobre un medicamento, DEBES usar la herramienta `get_information_about_medicine` para buscar la información. No intentes responder basándote en tu conocimiento general.
3.  **Basa tu respuesta en los hechos:** Formula tu respuesta final basándote *exclusivamente* en la información que te devuelve la herramienta en el `ToolMessage`.
4.  **Maneja la información faltante:** Si después de usar la herramienta no encuentras la información específica que el usuario solicita, responde claramente que no has podido encontrar esa información en el prospecto.
5.  **No des consejo médico:** Nunca ofrezcas consejo médico, diagnóstico o tratamiento. Tu función es solo transmitir la información del prospecto."""

def agent_node(state: AgentState, model: BaseChatModel):
    """The main ReAct brain. Formulates responses using context and tools."""
    # The System Prompt is added as a behavioral anchor.
    context = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    
    if state.get('summary'):
        context.append(SystemMessage(f"Resumen de la conversación:\n{state['summary']}"))

    # The sliding window of recent messages is passed without size concerns.
    context.extend(state['messages'])
    
    logging.info(f"--- Context sent to Agent ---:\n{context}\n--------------------")
    response = model.invoke(context)
    return {"messages": [response]}

# This prompt instructs the LLM on how to rewrite search queries for better retrieval.
QUERY_REWRITER_PROMPT = """Eres un experto en reescribir consultas de búsqueda para una base de datos vectorial de prospectos de medicamentos.
Tu tarea es convertir una consulta de búsqueda simple en una pregunta detallada y autocontenida, utilizando el historial de la conversación como contexto.

**Instrucciones:**
1.  Analiza el historial de la conversación y la consulta de búsqueda simple.
2.  Identifica el **tema principal de la consulta** (el medicamento y la información específica solicitada).
3.  Incorpora **detalles relevantes del historial** (como condiciones médicas, edad, etc., del paciente) que sean pertinentes para esa consulta.
4.  Ignora temas de preguntas anteriores que no estén relacionados con la consulta actual.
5.  El resultado debe ser una única pregunta clara y concisa, optimizada para la búsqueda semántica.

**Historial de Conversación (Resumen y últimos mensajes):**
{conversation_history}

**Consulta Simple a Reescribir:**
{query}

**Consulta Reescrita:**"""

def query_rewriter_node(state: AgentState, model: BaseChatModel):
    """
    Rewrites the agent's search query to include relevant context from the conversation.
    """
    logging.info("--- Rewriting query for search ---")
    
    # Extract the last tool call
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {} # Do nothing if there's no tool call

    original_tool_call = last_message.tool_calls[0]
    original_query = original_tool_call['args']['query']

    # Build the history for the prompt
    history = state.get('summary', '') + "\n\n"
    history += "\n".join([f"{type(m).__name__}: {m.content}" for m in state['messages'][:-1]])
    
    # Create and invoke the prompt
    rewriter_prompt = QUERY_REWRITER_PROMPT.format(
        conversation_history=history,
        query=original_query
    )
    rewritten_query = model.invoke(rewriter_prompt).content
    
    logging.info(f"--- Original Query: '{original_query}' ---")
    logging.info(f"--- Rewritten Query: '{rewritten_query}' ---")
    
    # Modify the tool call with the new query
    new_tool_calls = last_message.tool_calls.copy()
    new_tool_calls[0]['args']['query'] = rewritten_query
    
    # Create a new message to replace the old one
    new_message = AIMessage(
        content=last_message.content,
        tool_calls=new_tool_calls,
        id=last_message.id # Reuse the ID
    )
    
    # Replace the last message in the state
    all_but_last = state['messages'][:-1]
    return {"messages": all_but_last + [new_message]}


def conversational_node(state: AgentState):
    """Node for handling simple greetings and farewells."""
    logging.info("--- Executing conversational greeting node ---")
    return {"messages": [AIMessage(content="¡Hola! Soy un asistente médico virtual. ¿En qué puedo ayudarte?")]}

def unauthorized_question_node(state: AgentState, known_medicines: List[str]):
    """
    Node to handle questions about unknown medications.
    It informs the user and lists the available medicines.
    """
    logging.info("--- Executing unauthorized question node ---")
    
    known_medicines_str = ", ".join(med.title() for med in known_medicines)
    content = (
        f"Lo siento, no tengo información sobre el medicamento que mencionas. "
        f"Actualmente, solo puedo responder preguntas sobre: {known_medicines_str}."
    )
    
    return {"messages": [AIMessage(content=content)]}

def handle_retrieval_failure_node(state: AgentState):
    """
    Node to handle cases where the database search returns no documents.
    """
    logging.info("--- Executing retrieval failure node ---")
    
    last_known_medicine = state.get("current_medicines", [])[-1] if state.get("current_medicines") else "el medicamento"
    
    content = (
        f"He buscado en la información de '{last_known_medicine.title()}', pero no he podido encontrar una respuesta a tu pregunta específica. "
        "¿Hay algo más sobre este medicamento en lo que pueda ayudarte o quieres que intentemos con una pregunta diferente?"
    )
    
    return {"messages": [AIMessage(content=content)]}

def summarize_node(state: AgentState, model: BaseChatModel):
    """
    This node updates the summary and, crucially, resets the turn counter.
    """
    logging.info("--- Summarizing and pruning message window... ---")
    conversation_to_summarize = state['messages']
    
    summary_prompt = [
        SystemMessage(content="Resume la conversación médica. Preserva datos clave."),
        HumanMessage(content=(
            f"RESUMEN ACTUAL:\n{state.get('summary', 'No hay resumen previo.')}\n\n"
            "NUEVOS TURNOS A AÑADIR A ESTE RESUMEN:\n" +
            "\n".join(f"- {type(m).__name__}: {m.content}" for m in conversation_to_summarize)
        ))
    ]
    new_summary = model.invoke(summary_prompt).content
    
    # We now remove ALL messages from the short-term memory window.
    # The 'summary' becomes the single source of truth about the past, removing redundancy.
    messages_to_remove = [RemoveMessage(id=m.id) for m in state["messages"]]

    logging.info(f"--- New summary generated. Message window reset ({len(messages_to_remove)} messages removed). ---")
    # Reset the turn counter.
    return {"summary": new_summary, "messages": messages_to_remove, "turn_count": 0}
    
def end_of_turn_node(state: AgentState):
    """
    This node acts as the final checkpoint of a turn
    and is responsible for incrementing the turn counter.
    """
    logging.info("--- End of turn, updating counter. ---")
    return {"turn_count": state.get("turn_count", 0) + 1}

def pruning_node(state: AgentState):
    """
    Cleans the message history by removing intermediate tool-related messages
    (AIMessage with tool_calls and the corresponding ToolMessage) after the
    agent has produced its final response.
    """
    logging.info("--- Memory Pruning: Cleaning tool messages... ---")
    
    messages = state['messages']
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
    messages_to_remove = [RemoveMessage(id=msg_id) for msg_id in message_ids_to_remove]
    
    logging.info(f"--- Intermediate messages removed: {len(messages_to_remove)} ---")
    return {"messages": messages_to_remove}

# ==============================================================================
# === 5. CONDITIONAL LOGIC AND GRAPH CONSTRUCTION                            ===
# ==============================================================================
def route_after_router(state: AgentState) -> str:
    """Decides the path based on the intent. Most will go to the agent."""
    intent = state.get("intent")
    if intent == "saludo_despedida":
        return "conversational"
    if intent == "pregunta_no_autorizada":
        return "unauthorized"
    return "agent"

def should_continue_react(state: AgentState) -> str:
    """
    Decides if the agent should use a tool or if it has finished reasoning
    and should proceed to the final checkpoint (end_of_turn).
    """
    if isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls:
        return "tools"
    return "end_of_turn"

def route_after_tools(state: AgentState) -> str:
    """
    Decides the path after tool execution.
    If the search failed, it goes to a failure node.
    Otherwise, it returns to the agent to synthesize the response.
    """
    last_message = state['messages'][-1]
    if isinstance(last_message, ToolMessage) and last_message.content == RETRIEVAL_FAILURE_MESSAGE:
        logging.info("--- Retrieval failure detected. Rerouting to failure handler. ---")
        return "failure"
    
    logging.info("--- Retrieval successful. Returning to agent. ---")
    return "success"

def should_summarize_or_end(state: AgentState) -> str:
    """
    Decides if enough turns have passed to require a summary,
    based on the explicit 'turn_count'.
    """
    turn_count = state.get("turn_count", 0)
    logging.info(f"--- Summary check: Current turn {turn_count} ---")
    return "summarize" if turn_count >= SUMMARIZATION_TURN_THRESHOLD else "end"

# ==============================================================================
# === 6. GRAPH COMPILATION AND EXECUTION                                     ===
# ==============================================================================

# --- Graph Construction and Instantiation ---
# We move all the construction logic here so that the 'app' variable
# can be imported by LangGraph Studio and other UIs.
memory = MemorySaver()

print("Initializing clients and models...")
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
known_medicines = get_known_medicines(supabase)

embeddings = get_embeddings_model()
supabase_retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)

# Using a lambda for dependency injection is the most robust solution
# for it to work in both the console and LangGraph Studio.
medicine_tool = Tool(
    name="get_information_about_medicine",
    description="Busca en la BBDD de prospectos información sobre un medicamento.",
    func=lambda query: get_information_about_medicine(query, retriever=supabase_retriever),
    args_schema=MedicineToolInput
)

agent_llm = create_llm(config.AGENT_MODEL)
agent_llm_with_tools = agent_llm.bind_tools([medicine_tool])

router_llm = create_llm(config.ROUTER_MODEL)
router_llm_with_intent = router_llm.bind_tools([UserIntent])

tools = [medicine_tool]
tool_node = ToolNode(tools)

# --- Workflow Definition ---
workflow = StateGraph(AgentState)

# Partial function application to inject dependencies into the nodes
router_with_model_and_medicines = functools.partial(
    router_node, model=router_llm_with_intent, known_medicines=known_medicines
)
unauthorized_node_with_medicines = functools.partial(
    unauthorized_question_node, known_medicines=known_medicines
)
rewriter_node_with_model = functools.partial(query_rewriter_node, model=router_llm)

# Add nodes to the graph
workflow.add_node("router", router_with_model_and_medicines)
workflow.add_node("agent", functools.partial(agent_node, model=agent_llm_with_tools))
workflow.add_node("tools", tool_node)
workflow.add_node("query_rewriter", rewriter_node_with_model)
workflow.add_node("conversational", conversational_node)
workflow.add_node("unauthorized", unauthorized_node_with_medicines)
workflow.add_node("handle_retrieval_failure", handle_retrieval_failure_node)
workflow.add_node("summarizer", functools.partial(summarize_node, model=router_llm))
workflow.add_node("end_of_turn", end_of_turn_node)
workflow.add_node("pruning", pruning_node)
workflow.set_entry_point("router")

# --- Graph Edges ---
workflow.add_conditional_edges(
    "router", 
    route_after_router,
    {"agent": "agent", "conversational": "conversational", "unauthorized": "unauthorized"}
)
workflow.add_conditional_edges(
    "agent", 
    should_continue_react,
    {"tools": "query_rewriter", "end_of_turn": "pruning"}
)
workflow.add_conditional_edges(
    "end_of_turn",
    should_summarize_or_end,
    {"summarize": "summarizer", "end": END}
)
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "success": "agent",
        "failure": "handle_retrieval_failure"
    }
)

workflow.add_edge("query_rewriter", "tools")
workflow.add_edge("conversational", "end_of_turn")
workflow.add_edge("unauthorized", "end_of_turn")
workflow.add_edge("handle_retrieval_failure", "end_of_turn")
workflow.add_edge("pruning", "end_of_turn")
workflow.add_edge("summarizer", END)

# This version is compiled WITHOUT a checkpointer for LangGraph Studio.
app = workflow.compile()
print("Graph compiled successfully for LangGraph Studio.")

# This version is compiled WITH a memory checkpointer for use in the console or UI.
console_app = workflow.compile(checkpointer=memory)
print("Graph re-compiled with in-memory checkpointer for console and UI.")


if __name__ == '__main__':
    # This block is only executed when the script is run directly.
    # It handles the console interface.
    print("Checking environment variables...")
    config.check_env_vars()

    print("\nGenerating graph visualization...")
    try:
        # Render locally with Graphviz (robust and network-independent)
        png_bytes = console_app.get_graph().draw_png()
        with open("graph.png", "wb") as f: f.write(png_bytes)
        print("Graph image saved as 'graph.png'!")
    except Exception as e:
        print(f"\nCould not generate graph image: {e}")

    print("\n--- Medical Bot Initialized (V5 - Turn-Count Memory) ---")
    thread_id = str(uuid.uuid4())
    print(f"Starting new conversation with Thread ID: {thread_id}")
    
    run_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    while True:
        try:
            question = input("\nTu pregunta: ")
            if question.lower() in ['exit', 'quit']: break

            inputs = {"messages": [HumanMessage(content=question)]}
            
            print("\n--- Thinking... ---")
            # Use the console version for the interactive loop.
            for event in console_app.stream(inputs, config=run_config, stream_mode="values"):
                if "intent" in event and event["intent"]:
                    print(f"-> Intent: {event['intent']}")
                
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                     print(f"\nBot Response:\n{last_message.content}")

        except KeyboardInterrupt: break
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            break
    
    print("\nGoodbye. Take care!")