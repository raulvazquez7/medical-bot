import logging
import uuid
import functools
from typing import TypedDict, Annotated, List, Literal, Optional

# LangChain & LangGraph specific imports
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.language_models.chat_models import BaseChatModel
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
from src.app import SupabaseRetriever, format_docs_with_sources
from src.models import get_embeddings_model

# ==============================================================================
# === 1. DEFINICIÓN DEL ESTADO DEL AGENTE                                    ===
# ==============================================================================
class AgentState(TypedDict):
    """
    Representa el estado de nuestro agente.

    Attributes:
        messages: El historial completo de la conversación.
        summary: El resumen de la conversación.
        turn_count: Un contador para saber cuántos turnos han pasado.
        current_medicines: Lista de medicamentos identificados en la conversación.
        intent: La intención clasificada del último mensaje del usuario.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    turn_count: int
    current_medicines: List[str]
    intent: str

# ==============================================================================
# === 2. DEFINICIÓN DE HERRAMIENTAS (AGENTE Y ROUTER)                        ===
# ==============================================================================

class MedicineToolInput(BaseModel):
    query: str = Field(description="La pregunta específica sobre el medicamento a buscar.")

@tool
def get_information_about_medicine(query: str, retriever: SupabaseRetriever) -> str:
    """Busca en la BBDD de prospectos información sobre un medicamento."""
    logging.info(f"--- Ejecutando Herramienta de Búsqueda para: '{query}' ---")
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró información relevante en la base de datos para esa consulta."
    return format_docs_with_sources(docs)

class UserIntent(BaseModel):
    """Clasifica la intención del usuario para enrutar la conversación."""
    intent: Literal[
        "pregunta_medicamento", 
        "pregunta_general", 
        "saludo_despedida"
    ] = Field(description="La intención principal del mensaje del usuario.")
    
    medicine_name: Optional[str] = Field(
        description="El nombre del medicamento identificado en la pregunta, si lo hay."
    )

# ==============================================================================
# === 3. FUNCIONES DE INICIALIZACIÓN Y DATOS                                 ===
# ==============================================================================
def get_known_medicines(client: SupabaseClient) -> List[str]:
    """Consulta la BBDD para obtener la lista de medicamentos conocidos."""
    try:
        response = client.rpc('get_distinct_medicine_names', {}).execute()
        medicines = [item['medicine_name'].lower() for item in response.data]
        logging.info(f"Medicamentos cargados desde Supabase: {medicines}")
        return medicines
    except Exception as e:
        logging.error(f"No se pudo obtener la lista de medicamentos: {e}")
        return []

# ==============================================================================
# === 4. HELPER PARA CREACIÓN DE MODELOS                                     ===
# ==============================================================================

def create_llm(model_name: str) -> BaseChatModel:
    """
    [NUEVO] Crea una instancia del modelo de chat basándose en el nombre del config,
    centralizando la lógica de selección de proveedor (OpenAI vs. Google).
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
        raise ValueError(f"Modelo no soportado: {model_name}")

# ==============================================================================
# === 5. DEFINICIÓN DE LOS NODOS DEL GRAFO                                   ===
# ==============================================================================
SUMMARIZATION_THRESHOLD = 3

def router_node(state: AgentState, model: BaseChatModel):
    """El punto de entrada. Clasifica la intención del usuario para decidir la ruta."""
    logging.info("--- Router: Clasificando intención del usuario ---")
    last_message = state['messages'][-1].content
    
    prompt = (
        f"Clasifica la pregunta de un usuario. Las categorías son: 'pregunta_medicamento', "
        f"'saludo_despedida', o 'pregunta_general' (para cualquier otra cosa, como preguntas "
        f"personales, triviales, o de seguimiento). Extrae el nombre del medicamento "
        f"solo si la intención es 'pregunta_medicamento'.\n\n"
        f"Pregunta: '{last_message}'"
    )
    
    try:
        structured_response = model.invoke(prompt, tools=[UserIntent])
        if not structured_response.tool_calls:
            raise ValueError("El modelo no ha devuelto una intención estructurada.")
            
        intent_tool_call = structured_response.tool_calls[0]
        intent = intent_tool_call['args']['intent']
        medicine = intent_tool_call['args'].get('medicine_name', None)
        
        logging.info(f"--- Intención clasificada: {intent} | Medicamento: {medicine} ---")
        
        current_medicines = state.get("current_medicines", [])
        if medicine and medicine.lower() not in current_medicines:
            current_medicines.append(medicine.lower())
        
        return {"intent": intent, "current_medicines": current_medicines}

    except Exception as e:
        logging.error(f"Error en el router: {e}. Se usará 'pregunta_general' como fallback.")
        return {"intent": "pregunta_general"}

def agent_node(state: AgentState, model: BaseChatModel):
    """El nodo principal (cerebro ReAct). Formula respuestas usando el contexto y herramientas."""
    context = []
    if state.get("current_medicines"):
        context.append(SystemMessage(f"INFO: La conversación trata sobre: {', '.join(state['current_medicines'])}."))
    if state.get('summary'):
        context.append(SystemMessage(f"Resumen de la conversación:\n{state['summary']}"))

    last_human_idx = -1
    for i, msg in reversed(list(enumerate(state['messages']))):
        if isinstance(msg, HumanMessage):
            last_human_idx = i
            break
    context.extend(state['messages'][last_human_idx:])

    logging.info(f"--- Contexto enviado al Agente ---:\n{context}\n--------------------")
    response = model.invoke(context)
    return {"messages": [response]}

def conversational_node(state: AgentState):
    """Nodo para respuestas de saludo."""
    logging.info("--- Ejecutando nodo conversacional de saludo ---")
    return {"messages": [AIMessage(content="¡Hola! Soy un asistente médico virtual. ¿En qué puedo ayudarte?")]}

def summarize_node(state: AgentState, model: BaseChatModel):
    """Actualiza el resumen de la conversación."""
    logging.info("--- Resumiendo conversación... ---")
    conversation_to_summarize = state['messages'][-SUMMARIZATION_THRESHOLD * 2:]
    summary_prompt = [
        SystemMessage(content="Resume la conversación médica. Preserva datos clave."),
        HumanMessage(content=(
            f"RESUMEN ACTUAL:\n{state.get('summary', 'No hay resumen previo.')}\n\n"
            "NUEVOS TURNOS A AÑADIR:\n" +
            "\n".join(f"- {type(m).__name__}: {m.content}" for m in conversation_to_summarize)
        ))
    ]
    new_summary = model.invoke(summary_prompt).content
    logging.info(f"--- Nuevo resumen generado: {new_summary} ---")
    return {"summary": new_summary, "turn_count": 0}

def end_of_turn_node(state: AgentState):
    """Incrementa el contador de turnos al final de cada ciclo."""
    logging.info("--- Fin del turno, actualizando contador. ---")
    return {"turn_count": state.get("turn_count", 0) + 1}

# ==============================================================================
# === 6. LÓGICA CONDICIONAL Y CONSTRUCCIÓN DEL GRAFO                       ===
# ==============================================================================
def route_after_router(state: AgentState) -> str:
    """Decide la ruta basándose en la intención. La mayoría irán al agente."""
    intent = state.get("intent")
    if intent == "saludo_despedida":
        return "conversational"
    # Todas las demás intenciones ahora son manejadas por el agente principal
    return "agent"

def should_continue_react(state: AgentState) -> str:
    """Decide si el agente debe usar una herramienta o ha terminado el turno."""
    if isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls:
        return "tools"
    return "end_of_turn"

def should_summarize(state: AgentState) -> str:
    """Decide si es momento de resumir la conversación."""
    return "summarize" if state.get("turn_count", 0) >= SUMMARIZATION_THRESHOLD else "end"

# ==============================================================================
# === 7. COMPILACIÓN Y EJECUCIÓN DEL GRAFO                                   ===
# ==============================================================================
if __name__ == '__main__':
    print("Checking environment variables...")
    config.check_env_vars()
    memory = MemorySaver()

    print("Initializing clients and models...")
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    known_medicines = get_known_medicines(supabase)
    
    embeddings = get_embeddings_model()
    supabase_retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)

    medicine_tool = Tool(
        name="get_information_about_medicine",
        description="Busca en la BBDD de prospectos información sobre un medicamento.",
        func=functools.partial(get_information_about_medicine, retriever=supabase_retriever),
        args_schema=MedicineToolInput
    )

    # --- [CORREGIDO] Leemos los modelos desde el fichero de configuración ---
    agent_llm = create_llm(config.AGENT_MODEL)
    agent_llm_with_tools = agent_llm.bind_tools([medicine_tool])
    
    router_llm = create_llm(config.ROUTER_MODEL)
    router_llm_with_intent = router_llm.bind_tools([UserIntent])
    # --------------------------------------------------------------------

    tools = [medicine_tool]
    tool_node = ToolNode(tools)
    
    # --- Construcción del Grafo ---
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", functools.partial(router_node, model=router_llm_with_intent))
    workflow.add_node("agent", functools.partial(agent_node, model=agent_llm_with_tools))
    workflow.add_node("tools", tool_node)
    workflow.add_node("conversational", conversational_node)
    workflow.add_node("end_of_turn", end_of_turn_node)
    workflow.add_node("summarizer", functools.partial(summarize_node, model=router_llm))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router", 
        route_after_router,
        {"agent": "agent", "conversational": "conversational"}
    )
    workflow.add_conditional_edges("agent", should_continue_react, {"tools": "tools", "end_of_turn": "end_of_turn"})
    workflow.add_conditional_edges("end_of_turn", should_summarize, {"summarize": "summarizer", "end": END})
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("conversational", "end_of_turn")
    workflow.add_edge("summarizer", END)

    app = workflow.compile(checkpointer=memory)
    print("Graph compiled successfully with Simplified Router.")

    print("\nGenerando visualización del grafo...")
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f: f.write(png_bytes)
        print("¡Imagen del grafo guardada como 'graph.png'!")
    except Exception as e:
        print(f"\nNo se pudo generar la imagen del grafo: {e}")

    print("\n--- Medical Bot Initialized (V3 - Simplified Router) ---")
    thread_id = str(uuid.uuid4())
    print(f"Starting new conversation with Thread ID: {thread_id}")
    run_config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            question = input("\nTu pregunta: ")
            if question.lower() in ['exit', 'quit']: break

            inputs = {"messages": [HumanMessage(content=question)]}
            
            print("\n--- Pensando... ---")
            for event in app.stream(inputs, config=run_config, stream_mode="values"):
                if "intent" in event and event["intent"]:
                    print(f"-> Intención: {event['intent']}")
                
                # Imprimir la respuesta final del agente o del nodo conversacional
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                     print(f"\nRespuesta del Bot:\n{last_message.content}")

        except KeyboardInterrupt: break
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            break
    
    print("\nAdiós. ¡Cuídate!")