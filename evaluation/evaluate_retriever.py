import sys
import os
import json
import logging
from tqdm import tqdm
from typing import List
import cohere  # <-- Importar cohere
import time  # <-- Importar la librería time

# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from supabase import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.app import SupabaseRetriever
from src import config
from src.models import get_embeddings_model, get_known_medicines

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Evaluación ---
# Movidas a src/config.py para centralizar la configuración
GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"


def detect_medicines_in_question(question: str, known_medicines: List[str]) -> List[str]:
    """Detecta qué medicamentos conocidos se mencionan en la pregunta."""
    detected = []
    # Comprobamos cada medicamento conocido.
    # El caso de "sintron" está incluido en "sintron 4", por ejemplo.
    for med in known_medicines:
        # Buscamos el nombre base del medicamento en la pregunta (ignorando mayúsculas/minúsculas)
        # ej. buscar "sintron" en la pregunta.
        if med.split()[0].lower() in question.lower():
            detected.append(med)
    return detected


def rerank_with_cohere(query: str, docs: List, cohere_client: cohere.Client) -> List:
    """Re-ordena los documentos usando Cohere Rerank."""
    logging.info(f"Re-ranking {len(docs)} documentos con Cohere...")
    
    # El modelo de rerank necesita los textos de los documentos
    texts_to_rerank = [doc.page_content for doc in docs]
    
    try:
        # Hacemos la llamada a la API de rerank
        reranked_results = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts_to_rerank,
            top_n=config.EVAL_FINAL_K
        )
        
        # Mapeamos los resultados re-ordenados de vuelta a nuestros objetos Document
        reranked_docs = []
        for hit in reranked_results.results:
            reranked_docs.append(docs[hit.index])
            
        logging.info(f"Re-ranking completado. Devolviendo los {len(reranked_docs)} mejores documentos.")
        return reranked_docs
        
    except Exception as e:
        logging.error(f"Error durante el re-ranking con Cohere: {e}")
        # Si falla el re-ranker, devolvemos los 5 primeros originales como fallback
        return docs[:config.EVAL_FINAL_K]


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
        embeddings = get_embeddings_model()

        # El retriever ahora busca un K inicial más grande, leído desde config
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings, top_k=config.EVAL_INITIAL_K)
        # Inicializamos el cliente de Cohere
        cohere_client = cohere.Client(config.COHERE_API_KEY)
    except Exception as e:
        logging.error(f"Error al inicializar los componentes: {e}", exc_info=True)
        return

    # --- 3. Obtener medicamentos conocidos para el filtrado ---
    known_medicines = get_known_medicines(supabase)
    if not known_medicines:
        logging.warning("No se encontraron medicamentos en la base de datos. La evaluación se ejecutará sin filtros.")

    # --- 4. Ejecutar Evaluación ---
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_mrr = 0
    
    # Usamos tqdm para una bonita barra de progreso
    for item in tqdm(golden_dataset, desc="Evaluando preguntas"):
        question = item["question"]
        expected_paths = item["expected_paths"]

        # Detectamos los medicamentos en la pregunta para aplicar el filtro
        medicines_to_filter = detect_medicines_in_question(question, known_medicines)
        
        # Log para ver qué filtro se aplica
        if medicines_to_filter:
            logging.info(f"Pregunta: '{question[:80]}...' -> Filtro aplicado: {medicines_to_filter}")
        else:
            logging.info(f"Pregunta: '{question[:80]}...' -> Sin filtro.")
            
        # Obtenemos los documentos de nuestro retriever, pasando el filtro
        retrieved_docs = retriever._get_relevant_documents(
            question, 
            filter_on_medicines=medicines_to_filter
        )
        
        # Aplicamos el re-ranking solo si está activado en la configuración
        if config.EVAL_USE_RERANKER:
            final_docs = rerank_with_cohere(question, retrieved_docs, cohere_client)
        else:
            # Si el re-ranker está desactivado, simplemente tomamos los k_final primeros resultados
            logging.info(f"Re-ranking desactivado. Tomando los primeros {config.EVAL_FINAL_K} documentos.")
            final_docs = retrieved_docs[:config.EVAL_FINAL_K]
        
        # Ahora trabajamos con los documentos finales
        retrieved_paths = [doc.metadata.get('path', '') for doc in final_docs]
        
        # Calculamos las métricas para esta pregunta
        precision, recall, f1 = calculate_metrics(retrieved_paths, expected_paths)
        mrr = calculate_mrr(retrieved_paths, expected_paths)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_mrr += mrr

        # Pausa condicional: solo esperar si estamos usando el re-ranker y su API de prueba
        if config.EVAL_USE_RERANKER:
            time.sleep(6)

    # --- 5. Calcular y Mostrar Resultados Finales ---
    num_questions = len(golden_dataset)
    avg_precision = total_precision / num_questions if num_questions > 0 else 0
    avg_recall = total_recall / num_questions if num_questions > 0 else 0
    avg_f1 = total_f1 / num_questions if num_questions > 0 else 0
    avg_mrr = total_mrr / num_questions if num_questions > 0 else 0

    print("\n" + "="*50)
    print("        Resultados de la Evaluación del Retriever")
    print("="*50)
    print(f" Dataset: {GOLDEN_DATASET_PATH}")
    print(f" Modelo de Embedding: {config.EMBEDDINGS_MODEL}")
    print(f" Re-Ranker activado: {'Sí' if config.EVAL_USE_RERANKER else 'No'}")
    print(f" Número de preguntas evaluadas: {num_questions}")
    print(f" Valor de k inicial (retriever): {config.EVAL_INITIAL_K}")
    print(f" Valor de k final (evaluado): {config.EVAL_FINAL_K}")
    print("-"*50)
    print(f" F1-Score Promedio:            {avg_f1:.2%}")
    print(f" MRR (Mean Reciprocal Rank):   {avg_mrr:.4f}")
    print(f" Precisión Promedio @{config.EVAL_FINAL_K}:      {avg_precision:.2%}")
    print(f" Recall Promedio @{config.EVAL_FINAL_K}:         {avg_recall:.2%}")
    print("="*50)
    print("\nExplicación de Métricas:")
    print(" - F1-Score: Media armónica de Precisión y Recall. Mide el balance general.")
    print(" - MRR:      Calidad del ranking. Un valor alto significa que la primera respuesta correcta aparece pronto.")
    print(" - Precisión: De lo que se devuelve, cuánto es correcto. Mide la 'limpieza' de los resultados.")
    print(" - Recall:    De lo que debería devolverse, cuánto se encuentra. Mide la 'completitud' de los resultados.")

if __name__ == '__main__':
    run_retriever_evaluation() 