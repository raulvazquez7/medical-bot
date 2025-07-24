import sys
import os
# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import logging
from typing import List
from collections import defaultdict
from supabase import Client, create_client
from pydantic import BaseModel, Field
from langchain.schema import Document, BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src import config

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- El Retriever Personalizado para Supabase ---
class SupabaseRetriever(BaseRetriever):
    """
    Un retriever personalizado que busca documentos en Supabase
    usando la función `match_documents` que creamos.
    """
    supabase_client: Client
    embeddings_model: OpenAIEmbeddings
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Dada una consulta de usuario, la convierte en un embedding y busca
        en la base de datos los chunks más relevantes.
        """
        logging.info(f"Generando embedding para la consulta: '{query}'")
        query_embedding = self.embeddings_model.embed_query(query)
        
        logging.info(f"Buscando los {self.top_k} documentos más relevantes en Supabase...")
        
        response = self.supabase_client.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': self.top_k
            }
        ).execute()

        if response.data:
            logging.info(f"Se encontraron {len(response.data)} documentos.")
            return [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in response.data]
        
        logging.warning("No se encontraron documentos relevantes.")
        return []

# --- Structured Output ---
class AnswerWithSources(BaseModel):
    """Estructura de datos para la respuesta del LLM, incluyendo la respuesta y las fuentes."""
    answer: str = Field(
        description="La respuesta textual, clara y concisa, a la pregunta del usuario."
    )
    cited_sources: List[int] = Field(
        description="Una lista de los NÚMEROS de las fuentes [Fuente 1], [Fuente 2], etc., que se utilizaron para generar la respuesta."
    )

class QueryAnalysis(BaseModel):
    """
    Estructura de datos para el análisis de la pregunta del usuario.
    Indica si la pregunta es sobre un medicamento conocido.
    """
    medicine_name: str = Field(description="El nombre del medicamento extraído de la pregunta. Si no se menciona ninguno, el valor debe ser 'N/A'.")
    is_known: bool = Field(description="Es 'true' si el medicamento extraído está en la lista de medicamentos conocidos, de lo contrario es 'false'.")


# --- Plantillas de Prompt ---

TRIAGE_PROMPT_TEMPLATE = """
Tu única tarea es analizar la pregunta del usuario para identificar si menciona un medicamento de la lista proporcionada.

LISTA DE MEDICAMENTOS CONOCIDOS:
{known_medicines}

PREGUNTA DEL USUARIO:
{question}

Compara el medicamento que identifiques en la pregunta con la lista de medicamentos conocidos. Si no se menciona ningún medicamento o se menciona uno que no está en la lista, considera que no es conocido.

Rellena la estructura de datos requerida con tu análisis.
"""

RAG_PROMPT_TEMPLATE = """
**REGLA DE SEGURIDAD CRÍTICA: Eres un asistente informativo, NO un profesional médico. Tu única función es reportar con precisión lo que dice el texto del prospecto proporcionado. Está terminantemente prohibido dar consejos, opiniones personales o recomendaciones de cualquier tipo (p. ej., no digas 'deberías tomar', 'te recomiendo que', 'es seguro para ti'). Tu única tarea es resumir la información de las fuentes.**
**REGLA DE ALCANCE: Solo puedes responder preguntas sobre la información contenida en el contexto. Si la pregunta es sobre otro tema, o es un saludo, indica amablemente que solo puedes responder sobre la información de los prospectos.**

Eres un asistente experto en farmacología y tu única función es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado. Eres preciso, riguroso y nunca inventas información.
A continuación se presenta el contexto, dividido en fuentes numeradas:

CONTEXTO:
---------------------
{context}
---------------------

PREGUNTA: {question}

Analiza el contexto y la pregunta y rellena la estructura de datos requerida con tu respuesta y los números de las fuentes que has utilizado.
"""

def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Función auxiliar para formatear los documentos recuperados,
    anteponiendo un identificador de fuente a cada uno.
    """
    return "\n\n---\n\n".join(
        f"[Fuente {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )

def get_known_medicines(supabase_client: Client) -> List[str]:
    """
    Recupera una lista única de nombres de medicamentos de la base de datos.
    NOTA: Para una base de datos grande, este enfoque no es eficiente.
    Una mejor solución sería un endpoint o vista SQL dedicada.
    """
    logging.info("Recuperando la lista de medicamentos conocidos de la base de datos...")
    response = supabase_client.table('documents').select('metadata').execute()
    
    if not response.data:
        return []
    
    all_metadata = [item['metadata'] for item in response.data]
    known_medicines = sorted(list(set(
        meta['medicine_name'] for meta in all_metadata if meta and 'medicine_name' in meta
    )))
    logging.info(f"Medicamentos conocidos encontrados: {known_medicines}")
    return known_medicines


def run_chatbot():
    """Inicializa y ejecuta el bucle principal del chatbot de consola."""
    try:
        logging.info("Inicializando y comprobando configuración...")
        config.check_env_vars()
        
        logging.info("Inicializando clientes...")
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDINGS_MODEL)
        
        logging.info(f"Configurando el modelo de chat: {config.CHAT_MODEL_TO_USE}")
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        elif "gpt" in config.CHAT_MODEL_TO_USE:
            llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            raise ValueError(f"Modelo de chat no soportado: {config.CHAT_MODEL_TO_USE}. Revisa la configuración en 'src/config.py'.")

        # --- Paso 0: Obtener la lista de medicamentos conocidos ---
        known_medicines = get_known_medicines(supabase)
        if not known_medicines:
            logging.warning("No se encontraron medicamentos en la base de datos. El guardrail de alcance no podrá funcionar.")

        # --- Cadena de Triaje (Guardrail 1) ---
        triage_prompt = ChatPromptTemplate.from_template(TRIAGE_PROMPT_TEMPLATE)
        triage_llm = llm.with_structured_output(QueryAnalysis)
        triage_chain = triage_prompt | triage_llm

        # --- Cadena RAG Principal ---
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        structured_llm_rag = llm.with_structured_output(AnswerWithSources)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
            | rag_prompt
            | structured_llm_rag
        )

        rag_chain_with_sources = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        print("\n--- Medical Bot Iniciado ---")
        print("¡Hola! Estoy listo para responder tus preguntas sobre medicamentos.")
        print("Escribe 'salir' para terminar la conversación.")
        
        while True:
            question = input("\nTu pregunta: ")
            if question.lower() in ['salir', 'exit', 'quit']:
                print("Hasta luego. ¡Cuídate!")
                break
            
            # --- Guardrail 1: Ejecución del Triaje ---
            print("\nAnalizando la pregunta...")
            analysis_result = triage_chain.invoke({
                "known_medicines": ", ".join(known_medicines),
                "question": question
            })

            if not analysis_result.is_known:
                print("\nRespuesta del Bot:")
                if analysis_result.medicine_name != 'N/A':
                    print(f"Lo siento, no puedo responder sobre '{analysis_result.medicine_name}'.")
                
                print(f"Actualmente solo tengo información detallada sobre los siguientes medicamentos: {', '.join(known_medicines)}.")
                continue # Pasa a la siguiente pregunta sin ejecutar la cadena RAG

            # Si el análisis es correcto, procedemos con la cadena RAG
            print("\nBuscando en la base de datos y generando respuesta...")
            result = rag_chain_with_sources.invoke(question)
            
            # Accedemos a los atributos del objeto Pydantic
            answer_obj = result["answer"]
            answer_text = answer_obj.answer
            cited_indices = answer_obj.cited_sources
            
            print("\nRespuesta del Bot:")
            print(answer_text)
            
            if cited_indices:
                print("\n--- Fuentes Citadas ---")
                unique_cited_docs = {idx: result["context"][idx-1] for idx in sorted(set(cited_indices)) if idx <= len(result["context"])}
                
                for idx, doc in unique_cited_docs.items():
                    medicine_name = doc.metadata.get('medicine_name', 'Nombre no disponible')
                    path = doc.metadata.get('path', 'Ruta no disponible')
                    
                    print(f"\n[Fuente {idx}] La información se encuentra en el prospecto de '{medicine_name}'")
                    print(f"  Ruta: {path}")

    except ValueError as e:
        logging.error(f"Error de configuración: {e}")
    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado: {e}", exc_info=True)

if __name__ == '__main__':
    run_chatbot()