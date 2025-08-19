import sys
import os
import logging
from typing import List

# --- Explanatory Docstring ---
"""
EXPERIMENTAL SCRIPT FOR COMPARING CITATION METHODS

PURPOSE:
This script is not part of the main chatbot application. Its sole
purpose is to perform a visual and qualitative comparison between two
different methods of generating citations for a RAG system:

1. Our current method, which uses a language model to generate
   citations based on the context and the question.
2. ContextCite, a Python library that also uses a language model
   but employs a different, more rigorous attribution technique.

"""

# Add the root path so Python can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- ContextCite Dependencies ---
from context_cite import ContextCiter
import torch

# --- Components from our application ---
from supabase import Client, create_client
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.app import SupabaseRetriever, format_docs_with_sources, AnswerWithSources, rag_prompt_template
from src import config
from src.models import get_embeddings_model # <-- Standardized import

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_comparison():
    """
    Interactive script to visually compare our current citation method
    against the ContextCite library for the same question and context.
    """
    try:
        logging.info("Initializing components for comparison...")
        config.check_env_vars()
        
        # --- Common Initialization ---
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        # Use the project's standard embedding model for consistency
        embeddings = get_embeddings_model()
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)

        # --- System A: Our current RAG ---
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm_A = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            llm_A = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        
        structured_llm_rag = llm_A.with_structured_output(AnswerWithSources)
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
            | rag_prompt_template # Using the imported prompt from app.py
            | structured_llm_rag
        )

        # --- System B: ContextCite ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Switched to a high-quality model that does not require authentication
        model_name_B = "HuggingFaceH4/zephyr-7b-beta"
        logging.info(f"ContextCite will use the '{model_name_B}' model on device '{device}'")

        print("\n--- Citation Comparison Tool ---")
        print("Type 'exit' to end.")

        while True:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit']:
                print("Goodbye.")
                break

            # 1. GET CONTEXT (COMMON FOR BOTH SYSTEMS)
            print("\n" + "="*80)
            logging.info(f"Step 1: Fetching context for the question...")
            context_docs = retriever._get_relevant_documents(question)
            if not context_docs:
                print("No relevant documents found.")
                continue
            logging.info(f"Context retrieved. {len(context_docs)} documents found.")
            print("="*80)

            # 2. RUN AND DISPLAY SYSTEM A (OUR METHOD)
            print("\n\n" + "-"*30 + " SYSTEM A: Our Current Method " + "-"*29)
            try:
                result_A = rag_chain_from_docs.invoke({"context": context_docs, "question": question})
                print("\n>>> Generated Answer (System A):")
                print(result_A.answer)
                
                print("\n>>> Cited Sources (System A):")
                if result_A.cited_sources:
                    for idx in sorted(set(result_A.cited_sources)):
                        if idx <= len(context_docs):
                            doc = context_docs[idx-1]
                            medicine_name = doc.metadata.get('medicine_name', 'N/A')
                            path = doc.metadata.get('path', 'N/A')
                            print(f"  - [Source {idx}] Leaflet: '{medicine_name}', Path: '{path}'")
                else:
                    print("  - The model did not cite any sources.")

            except Exception as e:
                logging.error(f"Error in System A: {e}", exc_info=True)

            # 3. RUN AND DISPLAY SYSTEM B (CONTEXTCITE)
            print("\n\n" + "-"*30 + " SYSTEM B: ContextCite " + "-"*35)
            try:
                context_string = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
                logging.info("Generating response and attributions with ContextCite...")
                
                cc = ContextCiter.from_pretrained(
                    model_name_B,
                    context=context_string,
                    query=question,
                    device=device
                )
                
                print("\n>>> Generated Answer (System B):")
                print(cc.response)

                print("\n>>> Attributions (Top 3 most influential sentences according to ContextCite):")
                attributions = cc.get_attributions(as_dataframe=True, top_k=3)
                print(attributions)

            except Exception as e:
                logging.error(f"Error in System B: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Fatal error in the comparison script: {e}", exc_info=True)

if __name__ == '__main__':
    run_comparison() 