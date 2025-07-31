import sys
import os
import json
import logging
from tqdm import tqdm
from typing import List

# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Dependencias de Evaluación ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

# --- Componentes de nuestra aplicación ---
from supabase import Client, create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

# Importamos directamente la implementación del Retriever de nuestra app
from src.app import SupabaseRetriever, format_docs_with_sources, AnswerWithSources
from src import config

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Evaluación ---
GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"
K_VALUE = 5

# Copiamos la plantilla de prompt de app.py para asegurar que la evaluación es idéntica
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


def run_generation_evaluation():
    """
    Función principal que ejecuta la cadena RAG completa para el golden dataset
    y la evalúa usando RAGAS.
    """
    logging.info("--- Iniciando Evaluación de la Generación (RAGAS) ---")

    # --- 1. Cargar el Golden Dataset ---
    try:
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            # Solo necesitamos las preguntas para esta evaluación
            questions = [item["question"] for item in json.load(f)]
        logging.info(f"Se han cargado {len(questions)} preguntas del Golden Dataset.")
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo '{GOLDEN_DATASET_PATH}'.")
        return

    # --- 2. Inicializar la Cadena RAG Completa ---
    logging.info("Inicializando la cadena RAG completa...")
    try:
        config.check_env_vars()
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDINGS_MODEL)
        
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)

        # Recreamos la cadena RAG exactamente como en app.py
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings, top_k=K_VALUE)
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

    except Exception as e:
        logging.error(f"Error al inicializar la cadena RAG: {e}", exc_info=True)
        return
    
    # --- 3. Generar Respuestas para el Dataset ---
    logging.info("Generando respuestas para el dataset de evaluación. Esto puede tardar unos minutos...")
    results_for_ragas = []
    for q in tqdm(questions, desc="Generando respuestas RAG"):
        try:
            result = rag_chain_with_sources.invoke(q)
            answer_obj = result["answer"]
            
            # El formato que RAGAS espera: question, answer, contexts (plural)
            results_for_ragas.append({
                "question": q,
                "answer": answer_obj.answer,
                "contexts": [doc.page_content for doc in result["context"]],
            })
        except Exception as e:
            logging.error(f"Error generando respuesta para la pregunta '{q}': {e}")
            results_for_ragas.append({ "question": q, "answer": "", "contexts": [] })

    # --- 4. Evaluar con RAGAS ---
    if not results_for_ragas:
        logging.error("No se generaron resultados, no se puede ejecutar la evaluación RAGAS.")
        return
        
    logging.info("Preparando el dataset para RAGAS...")
    evaluation_dataset = Dataset.from_list(results_for_ragas)
    
    # Elegimos las métricas con las que evaluamos la generación
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        # context_precision, <-- Eliminado temporalmente
    ]
    
    logging.info("Ejecutando la evaluación con RAGAS. Esto puede tardar varios minutos más, ya que hace llamadas a un LLM para cada evaluación...")
    score = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics_to_evaluate,
    )
    
    # --- 5. Mostrar Resultados ---
    print("\n" + "="*50)
    print("      Resultados de la Evaluación de Generación (RAGAS)")
    print("="*50)
    print(score)
    print("="*50)
    print("\nExplicación de Métricas:")
    print(" - faithfulness:      Mide si la respuesta alucina. Un 1.0 es 100% basada en el contexto.")
    print(" - answer_relevancy:  Mide si la respuesta es relevante para la pregunta.")


if __name__ == '__main__':
    run_generation_evaluation() 