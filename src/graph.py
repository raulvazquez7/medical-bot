import logging
import operator
import uuid
import functools
from typing import TypedDict, Annotated, List

# LangChain & LangGraph specific imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
# [CORRECCIÓN DEFINITIVA] Importamos 'check' desde 'psycopg.pool'
from psycopg.pool import check as pool_check
from psycopg.rows import dict_row
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

# Project specific imports
from src import config
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import Client, create_client
# Hacemos accesibles las clases necesarias desde app.py y models.py
# (A largo plazo, podríamos refactorizar esto a un módulo de 'tools' o 'retrievers')
from src.app import SupabaseRetriever, format_docs_with_sources
from src.models import get_embeddings_model
import os
from langgraph.checkpoint.memory import MemorySaver

# ==============================================================================
# === 1. DEFINICIÓN DEL ESTADO DEL AGENTE                                    ===
# ==============================================================================
# Siguiendo el patrón ReAct, el estado se simplifica a solo la lista de mensajes.
# Toda la información (preguntas, llamadas a herramientas, resultados) se gestiona
# como mensajes en este historial.

class AgentState(TypedDict):
    """
    Representa el estado del agente ReAct.
    
    Attributes:
        messages: El historial de la conversación.
        summary: Un resumen incremental de la conversación para gestionar contextos largos.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str

# ==============================================================================
# === 2. DEFINICIÓN DE LAS HERRAMIENTAS DEL AGENTE                           ===
# ==============================================================================
# Cada función decorada con @tool se convierte en una herramienta que el LLM puede
# decidir llamar. La descripción en el docstring es CRUCIAL, ya que es lo que
# el LLM usa para saber qué hace la herramienta.

@tool
def get_information_about_medicine(query: str) -> str:
    """
    Busca en la base de datos de prospectos de medicamentos para encontrar información
    relevante a la pregunta del usuario. Úsala cuando el usuario pregunte por
    dosis, efectos secundarios, contraindicaciones, etc. de un medicamento.
    """
    logging.info(f"--- Ejecutando Herramienta de Búsqueda para: '{query}' ---")
    
    # Inicializamos los clientes necesarios dentro de la herramienta
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    embeddings = get_embeddings_model()
    retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)
    
    # Obtenemos los documentos
    docs = retriever.invoke(query)
    
    if not docs:
        return "No se encontró información relevante en la base de datos para esa consulta."
    
    # Formateamos la salida para que sea legible para el LLM
    return format_docs_with_sources(docs)

@tool
def conversational_response(query: str) -> str:
    """
    Proporciona una respuesta conversacional para saludos, despedidas o charla trivial.
    Úsala si la pregunta del usuario no está relacionada con medicamentos.
    """
    logging.info(f"--- Ejecutando Herramienta Conversacional ---")
    return "Responde amablemente al usuario. Si te preguntan tu nombre, di que eres un asistente médico."

# Lista de herramientas que el agente podrá usar
tools = [get_information_about_medicine, conversational_response]

# ==============================================================================
# === 3. DEFINICIÓN DEL MODELO Y LOS NODOS DEL GRAFO                        ===
# ==============================================================================

# [NUEVO] Función de ayuda para centralizar la creación del LLM
def get_llm_with_tools():
    """Inicializa el LLM según la configuración y le vincula las herramientas."""
    if "gemini" in config.CHAT_MODEL_TO_USE:
        llm = ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0
        )
    elif "gpt" in config.CHAT_MODEL_TO_USE:
        llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0
        )
    else:
        raise ValueError(f"Modelo de chat no soportado: {config.CHAT_MODEL_TO_USE}")
    
    model_with_tools = llm.bind_tools(tools)
    return llm, model_with_tools

# Constante para el umbral de resumen
CONTEXT_SUMMARIZATION_THRESHOLD = 6

def _messages_since_last_summary(messages: list[BaseMessage]) -> int:
    # Busca el primer SystemMessage (nuestro resumen) y cuenta desde ahí
    last_summary_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            last_summary_idx = i
            break
    return max(0, len(messages) - (last_summary_idx + 1))

def summarize_context_node(state: AgentState, model: BaseChatModel):
    msgs = state["messages"]
    # Calcula tramo incremental a resumir y preserva los últimos 2 mensajes (contexto inmediato)
    last_summary_idx = -1
    for i, m in enumerate(msgs):
        if isinstance(m, SystemMessage):
            last_summary_idx = i
            break
    to_keep = msgs[-2:] if len(msgs) >= 2 else msgs
    to_summarize = msgs[last_summary_idx + 1 : len(msgs) - len(to_keep)]
    if not to_summarize:
        return {}

    current_summary = state.get("summary", "")
    # Construye mensajes para el LLM (sin templates, directo)
    sys_msg = SystemMessage(content=(
        "Eres un experto resumiendo conversaciones médicas con precisión."
        " Actualiza el resumen previo con los nuevos turnos, preservando nombres de fármacos,"
        " dosis, contraindicaciones, síntomas y preguntas clave. Devuelve solo el resumen actualizado."
    ))
    human_msg = HumanMessage(content=(
        f"RESUMEN ACTUAL:\n{current_summary}\n\n"
        "NUEVOS TURNOS A AÑADIR:\n" +
        "\n".join(f"- {type(m).__name__}: {m.content}" for m in to_summarize)
    ))
    new_summary = model.invoke([sys_msg, human_msg]).content

    new_messages = [SystemMessage(content=f"Resumen hasta ahora: {new_summary}")] + to_keep
    return {"messages": new_messages, "summary": new_summary}


# El ToolNode es un nodo pre-construido de LangGraph que sabe cómo llamar
# a las herramientas que le pasemos.
tool_node = ToolNode(tools)

# La creación del modelo se moverá al bloque __main__ para poder inyectarla
# en el nodo de resumen.

def agent_node(state: AgentState, model: BaseChatModel):
    """
    El nodo principal que llama al LLM para que razone.
    """
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# ==============================================================================
# === 4. LÓGICA CONDICIONAL Y CONSTRUCCIÓN DEL GRAFO                       ===
# ==============================================================================

# [CORREGIDO] Esta es la lógica de enrutamiento correcta que diseñamos
def route_after_agent(state: AgentState) -> str:
    """
    Decide el siguiente paso después de que el agente haya respondido.
    1. Si ha llamado a una herramienta, vamos a la acción.
    2. Si no, comprobamos si la conversación necesita un resumen.
    3. Si no, terminamos el turno.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Comprobamos si el historial desde el último resumen ha superado el umbral
    messages_since_summary = _messages_since_last_summary(state["messages"])
    if messages_since_summary >= CONTEXT_SUMMARIZATION_THRESHOLD:
        return "summarize"
    
    return "end"

# ==============================================================================
# === 5. COMPILACIÓN Y EJECUCIÓN DEL GRAFO                                   ===
# ==============================================================================

if __name__ == '__main__':
    
    print("Checking environment variables...")
    config.check_env_vars()

    # [SOLUCIÓN DEFINITIVA] Usamos un Pool de Conexiones robusto y resiliente.
    use_mem = os.getenv("USE_INMEMORY_CHECKPOINTER", "false").lower() in ("1","true","yes")
    if use_mem:
        memory = MemorySaver()
        print("Using in-memory checkpointer (no Postgres).")
    else:
        # Pool Postgres estable (sin imports extras):
        from psycopg_pool import ConnectionPool
        pool = ConnectionPool(
            conninfo=config.POSTGRES_CONN_STR,
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
            max_lifetime=300,
            reconnect_timeout=5,
        )
        with pool:
            memory = PostgresSaver(pool)
            memory.setup()
            # compila y ejecuta dentro del with

    # Inicializamos el LLM y los nodos
    llm, model_with_tools = get_llm_with_tools()
    summarizer_with_model = functools.partial(summarize_context_node, model=llm)
    agent_with_model = functools.partial(agent_node, model=model_with_tools)

    # Construimos el grafo con la arquitectura correcta
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_with_model)
    workflow.add_node("action", tool_node)
    workflow.add_node("summarizer", summarizer_with_model)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"tools": "action", "summarize": "summarizer", "end": END}
    )
    workflow.add_edge("action", "agent")
    workflow.add_edge("summarizer", END)
    
    # Compilamos el grafo
    app = workflow.compile(checkpointer=memory)
    print("Graph compiled successfully.")

    # --- El resto del código (visualización y bucle de chat) va aquí dentro ---
    # ... (código para generar imagen y bucle de chat sin cambios)
    print("\nGenerando visualización del grafo...")
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("¡Imagen del grafo guardada como 'graph.png'!")
    except Exception as e:
        print(f"\nNo se pudo generar la imagen del grafo: {e}")

    print("\n--- Medical Bot Initialized (ReAct Version with Summarization) ---")
    print("Hello! I am ready to answer your questions.")
    thread_id_input = input("Enter a Thread ID to continue a conversation, or press Enter to start a new one: ").strip()
    
    if thread_id_input:
        thread_id = thread_id_input
        print(f"Continuing conversation with Thread ID: {thread_id}")
    else:
        thread_id = str(uuid.uuid4())
        print(f"Starting a new conversation with Thread ID: {thread_id}")

    run_config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit']:
                print("Goodbye. Take care!")
                break

            inputs = {"messages": [HumanMessage(content=question)]}
            
            print("\n--- Thinking... ---")
            for event in app.stream(inputs, config=run_config, stream_mode="values"):
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    print(f"Calling tool: {last_message.tool_calls[0]['name']} with args: {last_message.tool_calls[0]['args']}")
                elif isinstance(last_message, AIMessage):
                    print("\nBot's Response:")
                    print(last_message.content)

        except KeyboardInterrupt:
            print("\nGoodbye. Take care!")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            break
