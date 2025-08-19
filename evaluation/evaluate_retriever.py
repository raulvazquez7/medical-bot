import sys
import os
import json
import logging
from tqdm import tqdm
from typing import List
import cohere  # <-- Import cohere
import time    # <-- Import the time library

# Add the root path so Python can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from supabase import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.app import SupabaseRetriever
from src import config
from src.models import get_embeddings_model, get_known_medicines

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Evaluation Constants ---
# Moved to src/config.py to centralize configuration
GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"


def detect_medicines_in_question(question: str, known_medicines: List[str]) -> List[str]:
    """Detects which known medicines are mentioned in the question."""
    detected = []
    # We check each known medicine.
    # The case for "sintron" is included in "sintron 4", for example.
    for med in known_medicines:
        # We search for the base name of the medicine in the question (case-insensitive)
        # e.g., search for "sintron" in the question.
        if med.split()[0].lower() in question.lower():
            detected.append(med)
    return detected


def rerank_with_cohere(query: str, docs: List, cohere_client: cohere.Client) -> List:
    """Re-ranks the documents using Cohere Rerank."""
    logging.info(f"Re-ranking {len(docs)} documents with Cohere...")
    
    # The rerank model needs the texts of the documents
    texts_to_rerank = [doc.page_content for doc in docs]
    
    try:
        # We make the call to the rerank API
        reranked_results = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts_to_rerank,
            top_n=config.EVAL_FINAL_K
        )
        
        # We map the re-ranked results back to our Document objects
        reranked_docs = []
        for hit in reranked_results.results:
            reranked_docs.append(docs[hit.index])
            
        logging.info(f"Re-ranking completed. Returning the top {len(reranked_docs)} documents.")
        return reranked_docs
        
    except Exception as e:
        logging.error(f"Error during Cohere re-ranking: {e}")
        # If the re-ranker fails, we return the original top K as a fallback
        return docs[:config.EVAL_FINAL_K]


def calculate_metrics(retrieved_paths, expected_paths):
    """
    Calculates precision, recall, and F1-Score for a single query.
    """
    # Convert lists to sets for efficient intersection
    retrieved_set = set(retrieved_paths)
    expected_set = set(expected_paths)

    # Correct documents we have retrieved
    true_positives = len(retrieved_set.intersection(expected_set))
    
    # Precision@k: What proportion of the retrieved documents were relevant?
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    
    # Recall@k: What proportion of ALL relevant documents did we manage to retrieve?
    recall = true_positives / len(expected_set) if expected_set else 0

    # F1-Score: The harmonic mean of precision and recall.
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_mrr(retrieved_paths, expected_paths):
    """
    Calculates the Reciprocal Rank for a single query.
    Finds the rank of the first correct answer and returns 1/rank.
    """
    for i, path in enumerate(retrieved_paths):
        if path in expected_paths:
            return 1 / (i + 1)
    return 0 # If no correct answer is found

def run_retriever_evaluation():
    """
    Main function that loads the dataset, runs the retriever, and calculates the metrics.
    """
    logging.info("--- Starting Retriever Evaluation ---")

    # --- 1. Load the Golden Dataset ---
    try:
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            golden_dataset = json.load(f)
        logging.info(f"Golden Dataset with {len(golden_dataset)} questions has been loaded.")
    except FileNotFoundError:
        logging.error(f"Error: File '{GOLDEN_DATASET_PATH}' not found. Make sure the file exists.")
        return
    
    # --- 2. Initialize Components ---
    logging.info("Initializing clients and Supabase retriever...")
    try:
        config.check_env_vars()
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = get_embeddings_model()

        # The retriever now fetches a larger initial K, read from config
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings, top_k=config.EVAL_INITIAL_K)
        # Initialize the Cohere client
        cohere_client = cohere.Client(config.COHERE_API_KEY)
    except Exception as e:
        logging.error(f"Error initializing components: {e}", exc_info=True)
        return

    # --- 3. Get known medicines for filtering ---
    known_medicines = get_known_medicines(supabase)
    if not known_medicines:
        logging.warning("No medicines found in the database. The evaluation will run without filters.")

    # --- 4. Run Evaluation ---
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_mrr = 0
    
    # Use tqdm for a nice progress bar
    for item in tqdm(golden_dataset, desc="Evaluating questions"):
        question = item["question"]
        expected_paths = item["expected_paths"]

        # We detect the medicines in the question to apply the filter
        medicines_to_filter = detect_medicines_in_question(question, known_medicines)
        
        # Log to see which filter is applied
        if medicines_to_filter:
            logging.info(f"Question: '{question[:80]}...' -> Filter applied: {medicines_to_filter}")
        else:
            logging.info(f"Question: '{question[:80]}...' -> No filter.")
            
        # We get the documents from our retriever, passing the filter
        retrieved_docs = retriever._get_relevant_documents(
            question, 
            filter_on_medicines=medicines_to_filter
        )
        
        # Apply re-ranking only if it is enabled in the configuration
        if config.EVAL_USE_RERANKER:
            final_docs = rerank_with_cohere(question, retrieved_docs, cohere_client)
        else:
            # If the re-ranker is disabled, we simply take the first k_final results
            logging.info(f"Re-ranking disabled. Taking the top {config.EVAL_FINAL_K} documents.")
            final_docs = retrieved_docs[:config.EVAL_FINAL_K]
        
        # Now we work with the final documents
        retrieved_paths = [doc.metadata.get('path', '') for doc in final_docs]
        
        # Calculate the metrics for this question
        precision, recall, f1 = calculate_metrics(retrieved_paths, expected_paths)
        mrr = calculate_mrr(retrieved_paths, expected_paths)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_mrr += mrr

        # Conditional pause: only wait if we are using the re-ranker and its trial API
        if config.EVAL_USE_RERANKER:
            time.sleep(6)

    # --- 5. Calculate and Display Final Results ---
    num_questions = len(golden_dataset)
    avg_precision = total_precision / num_questions if num_questions > 0 else 0
    avg_recall = total_recall / num_questions if num_questions > 0 else 0
    avg_f1 = total_f1 / num_questions if num_questions > 0 else 0
    avg_mrr = total_mrr / num_questions if num_questions > 0 else 0

    print("\n" + "="*50)
    print("        Retriever Evaluation Results")
    print("="*50)
    print(f" Dataset: {GOLDEN_DATASET_PATH}")
    print(f" Embedding Model: {config.EMBEDDINGS_MODEL}")
    print(f" Re-Ranker enabled: {'Yes' if config.EVAL_USE_RERANKER else 'No'}")
    print(f" Number of questions evaluated: {num_questions}")
    print(f" Initial k value (retriever): {config.EVAL_INITIAL_K}")
    print(f" Final k value (evaluated): {config.EVAL_FINAL_K}")
    print("-"*50)
    print(f" Average F1-Score:            {avg_f1:.2%}")
    print(f" MRR (Mean Reciprocal Rank):   {avg_mrr:.4f}")
    print(f" Average Precision @{config.EVAL_FINAL_K}:      {avg_precision:.2%}")
    print(f" Average Recall @{config.EVAL_FINAL_K}:         {avg_recall:.2%}")
    print("="*50)
    print("\nMetric Explanations:")
    print(" - F1-Score: Harmonic mean of Precision and Recall. Measures the overall balance.")
    print(" - MRR:      Quality of the ranking. A high value means the first correct answer appears early.")
    print(" - Precision: Of what is returned, how much is correct. Measures the 'cleanliness' of the results.")
    print(" - Recall:    Of what should be returned, how much is found. Measures the 'completeness' of the results.")

if __name__ == '__main__':
    run_retriever_evaluation() 