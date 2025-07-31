import sys
import os
import logging
from typing import List

# --- Docstring Explicativo ---
"""
SCRIPT DE EXPERIMENTACIÓN Y COMPARACIÓN DE MÉTODOS DE CITADO

PROPÓSITO:
Este script no forma parte de la aplicación principal del chatbot. Su único
objetivo es realizar una comparación visual y cualitativa entre dos métodos
diferentes de generación de citas para un sistema RAG:

1. Nuestro método actual, que utiliza un modelo de lenguaje para generar
   citas basado en el contexto y la pregunta.
2. ContextCite, una librería de Python que utiliza un modelo de lenguaje
   para generar citas basado en el contexto y la pregunta.

"""

# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Dependencias de ContextCite ---
from context_cite import ContextCiter
import torch

# --- Componentes de nuestra aplicación ---
from supabase import Client, create_client
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from src.app import SupabaseRetriever, format_docs_with_sources, AnswerWithSources, RAG_PROMPT_TEMPLATE
from src import config

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_comparison():
    """
    Script interactivo para comparar visualmente nuestro método de citado actual
    contra la librería ContextCite para la misma pregunta y contexto.
    """
    try:
        logging.info("Inicializando componentes para la comparación...")
        config.check_env_vars()
        
        # --- Inicialización Común ---
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDINGS_MODEL)
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)

        # --- Sistema A: Nuestro RAG actual ---
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm_A = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            llm_A = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        structured_llm_rag = llm_A.with_structured_output(AnswerWithSources)
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
            | rag_prompt
            | structured_llm_rag
        )

        # --- Sistema B: ContextCite ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Cambiamos a un modelo de alta calidad que no requiere autenticación
        model_name_B = "HuggingFaceH4/zephyr-7b-beta"
        logging.info(f"ContextCite usará el modelo '{model_name_B}' en el dispositivo '{device}'")

        print("\n--- Herramienta de Comparación de Citado ---")
        print("Escribe 'salir' para terminar.")

        while True:
            question = input("\nTu pregunta: ")
            if question.lower() in ['salir', 'exit', 'quit']:
                print("Hasta luego.")
                break

            # 1. OBTENER CONTEXTO (COMÚN PARA AMBOS SISTEMAS)
            print("\n" + "="*80)
            logging.info(f"Paso 1: Obteniendo contexto para la pregunta...")
            context_docs = retriever._get_relevant_documents(question)
            if not context_docs:
                print("No se encontraron documentos relevantes.")
                continue
            logging.info(f"Contexto recuperado. {len(context_docs)} documentos encontrados.")
            print("="*80)

            # 2. EJECUTAR Y MOSTRAR SISTEMA A (NUESTRO MÉTODO)
            print("\n\n" + "-"*30 + " SISTEMA A: Nuestro Método Actual " + "-"*29)
            try:
                result_A = rag_chain_from_docs.invoke({"context": context_docs, "question": question})
                print("\n>>> Respuesta Generada (Sistema A):")
                print(result_A.answer)
                
                print("\n>>> Fuentes Citadas (Sistema A):")
                if result_A.cited_sources:
                    for idx in sorted(set(result_A.cited_sources)):
                        if idx <= len(context_docs):
                            doc = context_docs[idx-1]
                            medicine_name = doc.metadata.get('medicine_name', 'N/A')
                            path = doc.metadata.get('path', 'N/A')
                            print(f"  - [Fuente {idx}] Prospecto: '{medicine_name}', Ruta: '{path}'")
                else:
                    print("  - El modelo no ha citado ninguna fuente.")

            except Exception as e:
                logging.error(f"Error en el Sistema A: {e}", exc_info=True)

            # 3. EJECUTAR Y MOSTRAR SISTEMA B (CONTEXTCITE)
            print("\n\n" + "-"*30 + " SISTEMA B: ContextCite " + "-"*35)
            try:
                context_string = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
                logging.info("Generando respuesta y atribuciones con ContextCite...")
                
                cc = ContextCiter.from_pretrained(
                    model_name_B,
                    context=context_string,
                    query=question,
                    device=device
                )
                
                print("\n>>> Respuesta Generada (Sistema B):")
                print(cc.response)

                print("\n>>> Atribuciones (Top 3 frases más influyentes según ContextCite):")
                attributions = cc.get_attributions(as_dataframe=True, top_k=3)
                print(attributions)

            except Exception as e:
                logging.error(f"Error en el Sistema B: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Error fatal en el script de comparación: {e}", exc_info=True)

if __name__ == '__main__':
    run_comparison() 