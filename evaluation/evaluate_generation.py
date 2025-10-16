import sys
import os
import json
import logging
from tqdm import tqdm
from typing import List

# Add the root path so Python can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Evaluation Dependencies ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

# --- Components from our application ---
from supabase import Client, create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.schema import Document

# Import the retriever implementation and prompt directly from our app
from src.app import SupabaseRetriever, format_docs_with_sources, AnswerWithSources, rag_prompt_template
from src import config
from src.models.embeddings import get_embeddings_model

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Evaluation Constants ---
GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"
K_VALUE = 5


def run_generation_evaluation():
    """
    Main function that runs the complete RAG chain for the golden dataset
    and evaluates it using RAGAS.
    """
    logging.info("--- Starting Generation Evaluation (RAGAS) ---")

    # --- 1. Load the Golden Dataset ---
    try:
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            # We only need the questions for this evaluation
            questions = [item["question"] for item in json.load(f)]
        logging.info(f"{len(questions)} questions have been loaded from the Golden Dataset.")
    except FileNotFoundError:
        logging.error(f"Error: File '{GOLDEN_DATASET_PATH}' not found.")
        return

    # --- 2. Initialize the Complete RAG Chain ---
    logging.info("Initializing the complete RAG chain...")
    try:
        config.check_env_vars()
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        # Use our centralized factory to get the embeddings model
        embeddings = get_embeddings_model()
        
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)

        # Recreate the RAG chain exactly as in app.py
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings, top_k=K_VALUE)
        # We now use the imported prompt template
        structured_llm_rag = llm.with_structured_output(AnswerWithSources)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
            | rag_prompt_template  # <-- Using the imported template
            | structured_llm_rag
        )
        rag_chain_with_sources = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    except Exception as e:
        logging.error(f"Error initializing the RAG chain: {e}", exc_info=True)
        return
    
    # --- 3. Generate Answers for the Dataset ---
    logging.info("Generating answers for the evaluation dataset. This may take a few minutes...")
    results_for_ragas = []
    for q in tqdm(questions, desc="Generating RAG answers"):
        try:
            result = rag_chain_with_sources.invoke(q)
            answer_obj = result["answer"]
            
            # The format RAGAS expects: question, answer, contexts (plural)
            results_for_ragas.append({
                "question": q,
                "answer": answer_obj.answer,
                "contexts": [doc.page_content for doc in result["context"]],
            })
        except Exception as e:
            logging.error(f"Error generating answer for question '{q}': {e}")
            results_for_ragas.append({ "question": q, "answer": "", "contexts": [] })

    # --- 4. Evaluate with RAGAS ---
    if not results_for_ragas:
        logging.error("No results were generated, cannot run RAGAS evaluation.")
        return
        
    logging.info("Preparing the dataset for RAGAS...")
    evaluation_dataset = Dataset.from_list(results_for_ragas)
    
    # We choose the metrics with which to evaluate generation
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
    ]
    
    logging.info("Running evaluation with RAGAS. This may take several more minutes, as it makes LLM calls for each evaluation...")
    score = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics_to_evaluate,
    )
    
    # --- 5. Display Results ---
    print("\n" + "="*50)
    print("      Generation Evaluation Results (RAGAS)")
    print("="*50)
    print(score)
    print("="*50)
    print("\nMetric Explanations:")
    print(" - faithfulness:      Measures if the answer hallucinates. 1.0 is 100% based on the context.")
    print(" - answer_relevancy:  Measures if the answer is relevant to the question.")


if __name__ == '__main__':
    run_generation_evaluation() 