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
# [NUEVO] Importamos RunnableConfig
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
# [MODIFICADO] Las importaciones ahora apuntan a los nuevos módulos.
from src.database import SupabaseRetriever, get_known_medicines
from src.utils import format_docs_with_sources
from src.models import get_embeddings_model

# ==============================================================================
# === 1. DEFINICIÓN DEL ESTADO DEL AGENTE                                    ===
# ==============================================================================
class AgentState(TypedDict):
    """
    Representa el estado de nuestro agente.

    Attributes:
        messages: Una ventana deslizante de los mensajes más recientes.
        summary: El resumen de la conversación a largo plazo.
        turn_count: Un contador explícito de los turnos de conversación.
        current_medicines: Lista de medicamentos identificados.
        intent: La intención clasificada del último mensaje.
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

# [NUEVO] Constante para identificar de forma fiable un fallo en la recuperación.
RETRIEVAL_FAILURE_MESSAGE = "No se encontró información relevante en la base de datos para esa consulta."

# [MODIFICADO] Eliminamos el decorador @tool y cambiamos la firma de la función.
def get_information_about_medicine(query: str, retriever: SupabaseRetriever) -> str:
    """Busca en la BBDD de prospectos información sobre un medicamento."""
    logging.info(f"--- Ejecutando Herramienta de Búsqueda para: '{query}' ---")
    docs = retriever.invoke(query)
    if not docs:
        return RETRIEVAL_FAILURE_MESSAGE
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

# [ELIMINADO] La función 'get_known_medicines' se ha movido a 'src/database.py'.

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
SUMMARIZATION_TURN_THRESHOLD = 3

def router_node(state: AgentState, model: BaseChatModel, known_medicines: List[str]):
    """El punto de entrada. Clasifica la intención y valida el medicamento contra una lista conocida."""
    logging.info("--- Router: Clasificando intención del usuario ---")
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
            raise ValueError("El modelo no ha devuelto una intención estructurada.")
            
        intent_tool_call = structured_response.tool_calls[0]
        intent = intent_tool_call['args']['intent']
        medicine = intent_tool_call['args'].get('medicine_name', None)
        
        logging.info(f"--- Intención clasificada por LLM: {intent} | Medicamento extraído: {medicine} ---")
        
        current_medicines = state.get("current_medicines", [])
        
        # Verificamos el medicamento extraído contra nuestra lista de confianza.
        if medicine and intent == "pregunta_medicamento":
            medicine_lower = medicine.lower()
            
            # [NUEVO] Lógica de validación más estricta para evitar falsos positivos.
            # Buscamos una coincidencia de palabra completa (ej: 'nolotil' y no 'nolotilazo').
            matched_medicine = None
            for known_med in known_medicines:
                if re.search(r'\b' + re.escape(known_med) + r'\b', medicine_lower):
                    matched_medicine = known_med
                    break 

            if matched_medicine:
                # Si es un medicamento conocido, lo añadimos al estado si no estaba ya.
                logging.info(f"--- Medicamento '{matched_medicine}' validado. ---")
                if matched_medicine not in current_medicines:
                    current_medicines.append(matched_medicine)
            else:
                # Si el LLM extrajo algo pero no lo conocemos, cambiamos la intención.
                logging.warning(f"--- Medicamento '{medicine_lower}' no es conocido. Re-enrutando a pregunta_no_autorizada. ---")
                intent = "pregunta_no_autorizada"
        
        return {"intent": intent, "current_medicines": current_medicines}

    except Exception as e:
        logging.error(f"Error en el router: {e}. Se usará 'preg' como fallback.")
        return {"intent": "pregunta_general"}

AGENT_SYSTEM_PROMPT = """Eres un asistente médico virtual especializado en información de prospectos de medicamentos.

Tu objetivo principal es responder a las preguntas de los usuarios utilizando únicamente la información obtenida a través de la herramienta `get_information_about_medicine`.

Sigue estas reglas estrictamente:
1.  **Piensa paso a paso.** Antes de responder, analiza la pregunta del usuario.
2.  **Uso obligatorio de herramientas:** Si la pregunta es sobre un medicamento, DEBES usar la herramienta `get_information_about_medicine` para buscar la información. No intentes responder basándote en tu conocimiento general.
3.  **Basa tu respuesta en los hechos:** Formula tu respuesta final basándote *exclusivamente* en la información que te devuelve la herramienta en el `ToolMessage`.
4.  **Maneja la información faltante:** Si después de usar la herramienta no encuentras la información específica que el usuario solicita, responde claramente que no has podido encontrar esa información en el prospecto.
5.  **No des consejo médico:** Nunca ofrezcas consejo médico, diagnóstico o tratamiento. Tu función es solo transmitir la información del prospecto."""

def agent_node(state: AgentState, model: BaseChatModel):
    """El nodo principal (cerebro ReAct). Formula respuestas usando el contexto y herramientas."""
    # [MODIFICADO] Añadimos el System Prompt como ancla de comportamiento.
    context = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    
    if state.get('summary'):
        context.append(SystemMessage(f"Resumen de la conversación:\n{state['summary']}"))

    # [CLAVE] Ahora 'state['messages']' es una ventana de los últimos N mensajes,
    # por lo que podemos pasarla entera sin preocuparnos por el tamaño.
    context.extend(state['messages'])
    
    logging.info(f"--- Contexto enviado al Agente ---:\n{context}\n--------------------")
    response = model.invoke(context)
    return {"messages": [response]}

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
    [NUEVO] Reescribe la consulta de búsqueda del agente para incluir contexto relevante.
    """
    logging.info("--- Reescribiendo consulta para búsqueda ---")
    
    # Extraemos la última llamada a herramienta
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {} # No hacer nada si no hay llamada a herramienta

    original_tool_call = last_message.tool_calls[0]
    original_query = original_tool_call['args']['query']

    # Construimos el historial para el prompt
    history = state.get('summary', '') + "\n\n"
    history += "\n".join([f"{type(m).__name__}: {m.content}" for m in state['messages'][:-1]])
    
    # Creamos el prompt y lo invocamos
    rewriter_prompt = QUERY_REWRITER_PROMPT.format(
        conversation_history=history,
        query=original_query
    )
    rewritten_query = model.invoke(rewriter_prompt).content
    
    logging.info(f"--- Consulta Original: '{original_query}' ---")
    logging.info(f"--- Consulta Reescrita: '{rewritten_query}' ---")
    
    # Modificamos la llamada a herramienta con la nueva query
    new_tool_calls = last_message.tool_calls.copy()
    new_tool_calls[0]['args']['query'] = rewritten_query
    
    # Creamos un nuevo mensaje para reemplazar el anterior
    new_message = AIMessage(
        content=last_message.content,
        tool_calls=new_tool_calls,
        id=last_message.id # Reutilizamos el ID
    )
    
    # Reemplazamos el último mensaje en el estado
    all_but_last = state['messages'][:-1]
    return {"messages": all_but_last + [new_message]}


def conversational_node(state: AgentState):
    """Nodo para respuestas de saludo."""
    logging.info("--- Ejecutando nodo conversacional de saludo ---")
    return {"messages": [AIMessage(content="¡Hola! Soy un asistente médico virtual. ¿En qué puedo ayudarte?")]}

def unauthorized_question_node(state: AgentState, known_medicines: List[str]):
    """
    [NUEVO] Nodo para gestionar preguntas sobre medicamentos no conocidos.
    Informa al usuario y lista los medicamentos disponibles.
    """
    logging.info("--- Ejecutando nodo de pregunta no autorizada ---")
    
    known_medicines_str = ", ".join(med.title() for med in known_medicines)
    content = (
        f"Lo siento, no tengo información sobre el medicamento que mencionas. "
        f"Actualmente, solo puedo responder preguntas sobre: {known_medicines_str}."
    )
    
    return {"messages": [AIMessage(content=content)]}

def handle_retrieval_failure_node(state: AgentState):
    """
    [NUEVO] Nodo para gestionar el caso en que la búsqueda no devuelve documentos.
    """
    logging.info("--- Ejecutando nodo de fallo de recuperación ---")
    
    last_known_medicine = state.get("current_medicines", [])[-1] if state.get("current_medicines") else "el medicamento"
    
    content = (
        f"He buscado en la información de '{last_known_medicine.title()}', pero no he podido encontrar una respuesta a tu pregunta específica. "
        "¿Hay algo más sobre este medicamento en lo que pueda ayudarte o quieres que intentemos con una pregunta diferente?"
    )
    
    return {"messages": [AIMessage(content=content)]}

def summarize_node(state: AgentState, model: BaseChatModel):
    """
    Este nodo actualiza el resumen y, crucialmente, resetea el contador de turnos.
    """
    logging.info("--- Resumiendo y podando la ventana de mensajes... ---")
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
    
    # [MODIFICADO] Ahora eliminamos TODOS los mensajes de la ventana de memoria a corto plazo.
    # El 'summary' se convierte en la única fuente de verdad sobre el pasado, eliminando la redundancia.
    messages_to_remove = [RemoveMessage(id=m.id) for m in state["messages"]]

    logging.info(f"--- Nuevo resumen generado. Ventana de mensajes reseteada ({len(messages_to_remove)} mensajes eliminados). ---")
    # Reseteamos el contador de turnos.
    return {"summary": new_summary, "messages": messages_to_remove, "turn_count": 0}
    
def end_of_turn_node(state: AgentState):
    """
    Este nodo actúa como el punto de control final de un turno
    y es responsable de incrementar el contador de turnos.
    """
    logging.info("--- Fin del turno, actualizando contador. ---")
    return {"turn_count": state.get("turn_count", 0) + 1}

def pruning_node(state: AgentState):
    """
    [NUEVO] Limpia el historial de mensajes eliminando los mensajes intermedios
    relacionados con las herramientas (AIMessage con tool_calls y el ToolMessage
    correspondiente) después de que el agente haya producido su respuesta final.
    """
    logging.info("--- Poda de Memoria: Limpiando mensajes de herramientas... ---")
    
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
    
    logging.info(f"--- Mensajes intermedios eliminados: {len(messages_to_remove)} ---")
    return {"messages": messages_to_remove}

# ==============================================================================
# === 6. LÓGICA CONDICIONAL Y CONSTRUCCIÓN DEL GRAFO                       ===
# ==============================================================================
def route_after_router(state: AgentState) -> str:
    """Decide la ruta basándose en la intención. La mayoría irán al agente."""
    intent = state.get("intent")
    if intent == "saludo_despedida":
        return "conversational"
    # [NUEVO] Añadimos la nueva ruta para el guardrail.
    if intent == "pregunta_no_autorizada":
        return "unauthorized"
    # Todas las demás intenciones ahora son manejadas por el agente principal
    return "agent"

def should_continue_react(state: AgentState) -> str:
    """
    Decide si el agente debe usar una herramienta o si ha terminado de razonar
    y debe pasar al punto de control final (end_of_turn).
    """
    if isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls:
        return "tools"
    return "end_of_turn"

def route_after_tools(state: AgentState) -> str:
    """
    [NUEVO] Decide la ruta después de la ejecución de herramientas.
    Si la búsqueda no devolvió documentos, va a un nodo de fallo.
    De lo contrario, vuelve al agente para que sintetice la respuesta.
    """
    last_message = state['messages'][-1]
    if isinstance(last_message, ToolMessage) and last_message.content == RETRIEVAL_FAILURE_MESSAGE:
        logging.info("--- Fallo de recuperación detectado. Re-enrutando al manejador de fallos. ---")
        return "failure"
    
    logging.info("--- Recuperación exitosa. Volviendo al agente. ---")
    return "success"

def should_summarize_or_end(state: AgentState) -> str:
    """
    [MODIFICADO] Decide si han pasado suficientes turnos para necesitar un resumen,
    basándose en el 'turn_count' explícito.
    """
    turn_count = state.get("turn_count", 0)
    logging.info(f"--- Chequeo de resumen: Turno actual {turn_count} ---")
    return "summarize" if turn_count >= SUMMARIZATION_TURN_THRESHOLD else "end"

# ==============================================================================
# === 7. COMPILACIÓN Y EJECUCIÓN DEL GRAFO                                   ===
# ==============================================================================

# --- Construcción e Instanciación del Grafo ---
# Movemos toda la lógica de construcción aquí para que la variable 'app'
# sea importable por LangGraph Studio.
memory = MemorySaver()

print("Initializing clients and models...")
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
known_medicines = get_known_medicines(supabase)

embeddings = get_embeddings_model()
supabase_retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)

# [CORREGIDO] Volvemos al patrón de 'lambda' para la inyección de dependencias.
# Es la solución más robusta para que funcione tanto en consola como en el Studio.
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

# [MODIFICADO] Grafo reestructurado para la nueva lógica de memoria
workflow = StateGraph(AgentState)

router_with_model_and_medicines = functools.partial(
    router_node, model=router_llm_with_intent, known_medicines=known_medicines
)
unauthorized_node_with_medicines = functools.partial(
    unauthorized_question_node, known_medicines=known_medicines
)
rewriter_node_with_model = functools.partial(query_rewriter_node, model=router_llm)

workflow.add_node("router", router_with_model_and_medicines)
workflow.add_node("agent", functools.partial(agent_node, model=agent_llm_with_tools))
workflow.add_node("tools", tool_node)
workflow.add_node("query_rewriter", rewriter_node_with_model)
workflow.add_node("conversational", conversational_node)
workflow.add_node("unauthorized", unauthorized_node_with_medicines)
workflow.add_node("handle_retrieval_failure", handle_retrieval_failure_node) # <-- [NUEVO]
workflow.add_node("summarizer", functools.partial(summarize_node, model=router_llm))
workflow.add_node("end_of_turn", end_of_turn_node)
workflow.add_node("pruning", pruning_node)
workflow.set_entry_point("router")

# --- Flujo de Grafo Actualizado ---
workflow.add_conditional_edges(
    "router", 
    route_after_router,
    {"agent": "agent", "conversational": "conversational", "unauthorized": "unauthorized"}
)
workflow.add_conditional_edges(
    "agent", 
    should_continue_react,
    {"tools": "query_rewriter", "end_of_turn": "pruning"} # [MODIFICADO] La salida a herramientas ahora va al rewriter
)
workflow.add_conditional_edges(
    "end_of_turn",
    should_summarize_or_end, # La decisión final se toma desde el contador de turnos
    {"summarize": "summarizer", "end": END}
)

# [NUEVO] Flujo condicional después de la ejecución de herramientas.
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
workflow.add_edge("handle_retrieval_failure", "end_of_turn") # <-- [NUEVO]
workflow.add_edge("pruning", "end_of_turn")
workflow.add_edge("summarizer", END)

# La variable 'app' ahora es global y está compilada SIN checkpointer para LangGraph Studio.
app = workflow.compile()
print("Graph compiled successfully for LangGraph Studio.")


if __name__ == '__main__':
    # El bloque __main__ ahora solo se encarga de la interfaz de consola.
    print("Checking environment variables...")
    config.check_env_vars()

    # [CORREGIDO] En lugar de usar '.with_checkpointer()', que ya no existe,
    # compilamos el 'workflow' original una segunda vez, ahora sí,
    # añadiendo el checkpointer de memoria para la consola.
    console_app = workflow.compile(checkpointer=memory)
    print("Graph re-compiled with in-memory checkpointer for console.")

    print("\nGenerando visualización del grafo...")
    try:
        # Render local con Graphviz (robusto y sin red)
        png_bytes = console_app.get_graph().draw_png()
        with open("graph.png", "wb") as f: f.write(png_bytes)
        print("¡Imagen del grafo guardada como 'graph.png'!")
    except Exception as e:
        print(f"\nNo se pudo generar la imagen del grafo: {e}")

    print("\n--- Medical Bot Initialized (V5 - Turn-Count Memory) ---")
    thread_id = str(uuid.uuid4())
    print(f"Starting new conversation with Thread ID: {thread_id}")
    
    # [MODIFICADO] El retriever ya no necesita pasarse en el config.
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
            
            print("\n--- Pensando... ---")
            # Usamos la versión de consola para el bucle interactivo.
            for event in console_app.stream(inputs, config=run_config, stream_mode="values"):
                if "intent" in event and event["intent"]:
                    print(f"-> Intención: {event['intent']}")
                
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                     print(f"\nRespuesta del Bot:\n{last_message.content}")

        except KeyboardInterrupt: break
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            break
    
    print("\nAdiós. ¡Cuídate!")