import sys
import os
import json
import logging
from tqdm import tqdm

# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from supabase import Client, create_client
from langchain_openai import OpenAIEmbeddings

from src.app import SupabaseRetriever
from src import config

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Evaluación ---
GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"
# El valor de 'k' para nuestras métricas. Coincide con el top_k de nuestro retriever en app.py
K_VALUE = 5 

def calculate_metrics(retrieved_paths, expected_paths):
    """
    Calcula la precisión, el recall y el F1-Score para una sola consulta.
    """
    # Convertimos las listas a conjuntos para una intersección eficiente
    retrieved_set = set(retrieved_paths)
    expected_set = set(expected_paths)

    # Documentos correctos que hemos recuperado
    true_positives = len(retrieved_set.intersection(expected_set))
    
    # Precision@k: ¿Qué proporción de los documentos recuperados eran relevantes?
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    
    # Recall@k: ¿Qué proporción de TODOS los documentos relevantes conseguimos recuperar?
    recall = true_positives / len(expected_set) if expected_set else 0

    # F1-Score: La media armónica de precisión y recall.
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_mrr(retrieved_paths, expected_paths):
    """
    Calcula el Reciprocal Rank para una sola consulta.
    Encuentra el rango de la primera respuesta correcta y devuelve 1/rango.
    """
    for i, path in enumerate(retrieved_paths):
        if path in expected_paths:
            return 1 / (i + 1)
    return 0 # Si no se encuentra ninguna respuesta correcta


def run_retriever_evaluation():
    """
    Función principal que carga el dataset, ejecuta el retriever y calcula las métricas.
    """
    logging.info("--- Iniciando Evaluación del Retriever ---")

    # --- 1. Cargar el Golden Dataset ---
    try:
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            golden_dataset = json.load(f)
        logging.info(f"Se ha cargado el Golden Dataset con {len(golden_dataset)} preguntas.")
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo '{GOLDEN_DATASET_PATH}'. Asegúrate de que el archivo existe.")
        return
    
    # --- 2. Inicializar Componentes ---
    logging.info("Inicializando clientes y el retriever de Supabase...")
    try:
        config.check_env_vars()
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDINGS_MODEL)
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings, top_k=K_VALUE)
    except Exception as e:
        logging.error(f"Error al inicializar los componentes: {e}", exc_info=True)
        return

    # --- 3. Ejecutar Evaluación ---
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_mrr = 0
    
    # Usamos tqdm para una bonita barra de progreso
    for item in tqdm(golden_dataset, desc="Evaluando preguntas"):
        question = item["question"]
        expected_paths = item["expected_paths"]
        
        # Obtenemos los documentos de nuestro retriever
        retrieved_docs = retriever._get_relevant_documents(question)
        retrieved_paths = [doc.metadata.get('path', '') for doc in retrieved_docs]
        
        # Calculamos las métricas para esta pregunta
        precision, recall, f1 = calculate_metrics(retrieved_paths, expected_paths)
        mrr = calculate_mrr(retrieved_paths, expected_paths)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_mrr += mrr

    # --- 4. Calcular y Mostrar Resultados Finales ---
    num_questions = len(golden_dataset)
    avg_precision = total_precision / num_questions if num_questions > 0 else 0
    avg_recall = total_recall / num_questions if num_questions > 0 else 0
    avg_f1 = total_f1 / num_questions if num_questions > 0 else 0
    avg_mrr = total_mrr / num_questions if num_questions > 0 else 0

    print("\n" + "="*50)
    print("        Resultados de la Evaluación del Retriever")
    print("="*50)
    print(f" Dataset: {GOLDEN_DATASET_PATH}")
    print(f" Número de preguntas evaluadas: {num_questions}")
    print(f" Valor de k (documentos recuperados): {K_VALUE}")
    print("-"*50)
    print(f" F1-Score Promedio:            {avg_f1:.2%}")
    print(f" MRR (Mean Reciprocal Rank):   {avg_mrr:.4f}")
    print(f" Precisión Promedio @{K_VALUE}:      {avg_precision:.2%}")
    print(f" Recall Promedio @{K_VALUE}:         {avg_recall:.2%}")
    print("="*50)
    print("\nExplicación de Métricas:")
    print(" - F1-Score: Media armónica de Precisión y Recall. Mide el balance general.")
    print(" - MRR:      Calidad del ranking. Un valor alto significa que la primera respuesta correcta aparece pronto.")
    print(" - Precisión: De lo que se devuelve, cuánto es correcto. Mide la 'limpieza' de los resultados.")
    print(" - Recall:    De lo que debería devolverse, cuánto se encuentra. Mide la 'completitud' de los resultados.")

if __name__ == '__main__':
    run_retriever_evaluation() 