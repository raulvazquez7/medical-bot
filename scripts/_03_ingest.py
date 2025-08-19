import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import logging  
from supabase import create_client, Client
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from functools import partial
from langchain_core.documents import Document
from langchain.text_splitter import NLTKTextSplitter

# Import central configuration and functions from other scripts
from src import config
from src.models import get_embeddings_model
from scripts._01_pdf_to_markdown import generate_markdown_from_pdf_images
from scripts._02_markdown_to_chunks import markdown_to_semantic_blocks, create_sentence_window_chunks

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_medicine_name(markdown_text: str, fallback_filename: str) -> str:
    """
    Extracts the medicine name from the Markdown text.
    It looks for the text between 'Prospecto: información para el paciente' and 'Lea todo el prospecto...'.
    If not found, it uses the filename as a fallback.
    """
    try:
        # Use a regular expression to find the medicine name
        pattern = r"Prospecto: información para el paciente\s*\n\s*(.*?)\s*\n\s*Lea todo el prospecto"
        match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean up the result from potential newlines and extra spaces
            medicine_name = ' '.join(match.group(1).split()).strip()
            logging.info(f"Extracted medicine name: '{medicine_name}'")
            return medicine_name
    except Exception:
        pass  # If there is any error in the regex, use the fallback

    # If the pattern is not found, use the filename as a fallback
    name_from_file = os.path.splitext(fallback_filename)[0].replace('_', ' ').replace('-', ' ')
    logging.warning(f"Could not extract medicine name from prospectus. Using fallback: '{name_from_file}'")
    return name_from_file


def standardize_medicine_name(raw_name: str) -> str:
    """
    Normalizes a medicine name to a simple, canonical version.
    E.g., "Espidifen 600 mg granulado..." -> "espidifen"
          "ibuprofeno_cinfa_600" -> "ibuprofeno cinfa"
    """
    name = raw_name.lower()
    
    # Special cases first
    if "espidifen" in name:
        return "espidifen"
    if "nolotil" in name:
        return "nolotil"
    if "sintrom" in name:
        return "sintrom"
    if "lexatin" in name:
        return "lexatin"
    
    # More general cases
    if "ibuprofeno" in name and "cinfa" in name:
        return "ibuprofeno cinfa"
    if "ibuprofeno" in name and "kern" in name: # Example for future expansion
        return "ibuprofeno kern"

    # If it doesn't match any case, clean the base name from the file
    base_name = os.path.splitext(raw_name)[0]
    return base_name.replace('_', ' ').replace('-', ' ').strip()


def run_pipeline(pdf_filename: str, supabase_client: Client, embeddings_model: Embeddings, force_reparse: bool = False):
    """
    Orchestrates the complete pipeline for a single PDF file:
    1. Converts the PDF to Markdown.
    2. Creates semantic chunks from the Markdown.
    3. Generates embeddings and ingests them into Supabase.
    """
    logging.info(f"--- Starting pipeline for file: {pdf_filename} ---")
    
    # --- STEP 1: PDF to Markdown Conversion (with cache) ---
    logging.info("Step 1: Preparing PDF to Markdown conversion...")
    pdf_file_path = os.path.join(config.DATA_PATH, pdf_filename)
    if not os.path.exists(pdf_file_path):
        logging.error(f"PDF file '{pdf_file_path}' not found. Aborting.")
        return

    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-')
    md_filename = f"parsed_by_{model_name_slug}_{pdf_filename.replace('.pdf', '.md')}"
    md_file_path = os.path.join(config.MARKDOWN_PATH, md_filename)

    # Cache logic: only process if the file does not exist or if re-parsing is forced
    if not force_reparse and os.path.exists(md_file_path):
        logging.info(f"Markdown file '{md_file_path}' already exists. Skipping PDF parsing step.")
    else:
        logging.info("Starting PDF to Markdown conversion...")
        # Ensure the output directory for markdown exists
        if not os.path.exists(config.MARKDOWN_PATH):
            os.makedirs(config.MARKDOWN_PATH)

        generate_markdown_from_pdf_images(pdf_file_path, md_file_path)
        logging.info(f"PDF successfully converted to Markdown. File saved at: {md_file_path}")

    # --- STEP 2: Chunk Creation ---
    logging.info("Step 2: Creating semantic chunks from the Markdown file...")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Extract and then standardize the medicine name
    raw_medicine_name = extract_medicine_name(markdown_text, pdf_filename)
    medicine_name = standardize_medicine_name(raw_medicine_name)
    logging.info(f"Medicine name standardized to: '{medicine_name}'")

    semantic_blocks = markdown_to_semantic_blocks(markdown_text)
    
    # Switch the chunking function to our new Sentence-Window implementation
    chunks = create_sentence_window_chunks(
        blocks=semantic_blocks, 
        source_file=md_filename, 
        medicine_name=medicine_name
    )
    logging.info(f"{len(chunks)} (sentence-window) chunks were created for this document.")

    if not chunks:
        logging.warning("No chunks were generated. Aborting ingestion process.")
        return

    # --- STEP 3: Embedding Generation ---
    logging.info("Step 3: Generating embeddings...")
    texts_to_embed = [doc.page_content for doc in chunks]
    try:
        embeddings_list = embeddings_model.embed_documents(texts_to_embed)
        logging.info(f"{len(embeddings_list)} embeddings were generated.")
    except Exception as e:
        logging.error(f"Error generating embeddings for {md_filename}: {e}", exc_info=True)
        return

    # --- STEP 4: Clean Up Old Data in Supabase ---
    logging.info(f"Step 4: Cleaning up old records for '{md_filename}' in Supabase...")
    try:
        # We use the 'source' field in the metadata to identify and delete old chunks.
        # The '->>' operator extracts the JSON value as text for an exact comparison.
        supabase_client.table('documents').delete().eq('metadata->>source', md_filename).execute()
        logging.info("Cleanup of old records completed successfully.")
    except Exception as e:
        # We do not stop the process if this fails, as it might be the first time ingesting.
        logging.warning(f"Warning during cleanup of old data: {e}")

    # --- STEP 5: Ingest New Data into Supabase ---
    logging.info(f"Step 5: Preparing and uploading {len(chunks)} new records to Supabase...")
    records_to_insert = []
    for i, chunk in enumerate(chunks):
        records_to_insert.append({
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'embedding': embeddings_list[i]
        })

    try:
        supabase_client.table('documents').insert(records_to_insert).execute()
        logging.info(f"Success! {len(records_to_insert)} new records from {pdf_filename} have been inserted into the database.")
    except Exception as e:
        logging.error(f"Error inserting data for {pdf_filename}: {e}", exc_info=True)

    logging.info(f"--- Pipeline for {pdf_filename} finished. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processes and ingests a medicine leaflet in PDF format into the database.")
    parser.add_argument(
        "pdf_filename", 
        type=str, 
        help="The name of the PDF file to process (e.g., 'nolotil_575.pdf'). It must be in the 'data' folder."
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Forces the re-conversion from PDF to Markdown even if the file already exists."
    )
    args = parser.parse_args()

    try:
        # --- Client Initialization ---
        logging.info("Initializing and checking configuration...")
        config.check_env_vars() # Check that environment variables exist
        
        logging.info("Initializing Supabase and Embeddings clients...")
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = get_embeddings_model()
        
        # --- Run the Pipeline ---
        run_pipeline(args.pdf_filename, supabase, embeddings, args.force_reparse)

    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 