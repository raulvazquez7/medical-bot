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
from langchain.document_loaders import Document
from langchain.text_splitter import NLTKTextSplitter

# Importamos la configuración central y las funciones de los otros scripts
from src import config
from src.models import get_embeddings_model
from scripts._01_pdf_to_markdown import generate_markdown_from_pdf_images
from scripts._02_markdown_to_chunks import markdown_to_semantic_blocks, create_sentence_window_chunks

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_medicine_name(markdown_text: str, fallback_filename: str) -> str:
    """
    Extrae el nombre del medicamento del texto Markdown.
    Busca el texto entre 'Prospecto: información para el paciente' y 'Lea todo el prospecto...'.
    Si no lo encuentra, usa el nombre del archivo como alternativa.
    """
    try:
        # Usamos una expresión regular para encontrar el nombre del medicamento
        pattern = r"Prospecto: información para el paciente\s*\n\s*(.*?)\s*\n\s*Lea todo el prospecto"
        match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
        if match:
            # Limpiamos el resultado de posibles saltos de línea y espacios extra
            medicine_name = ' '.join(match.group(1).split()).strip()
            logging.info(f"Nombre del medicamento extraído: '{medicine_name}'")
            return medicine_name
    except Exception:
        pass  # Si hay cualquier error en la regex, usamos el fallback

    # Si no se encuentra el patrón, usamos el nombre del archivo como fallback
    name_from_file = os.path.splitext(fallback_filename)[0].replace('_', ' ').replace('-', ' ')
    logging.warning(f"No se pudo extraer el nombre del prospecto. Usando fallback: '{name_from_file}'")
    return name_from_file


def standardize_medicine_name(raw_name: str) -> str:
    """
    Normaliza el nombre de un medicamento a una versión canónica y simple.
    Ej: "Espidifen 600 mg granulado..." -> "espidifen"
        "ibuprofeno_cinfa_600" -> "ibuprofeno cinfa"
    """
    name = raw_name.lower()
    
    # Casos especiales primero
    if "espidifen" in name:
        return "espidifen"
    if "nolotil" in name:
        return "nolotil"
    if "sintrom" in name:
        return "sintrom"
    if "lexatin" in name:
        return "lexatin"
    
    # Casos más generales
    if "ibuprofeno" in name and "cinfa" in name:
        return "ibuprofeno cinfa"
    if "ibuprofeno" in name and "kern" in name: # Ejemplo para futura expansión
        return "ibuprofeno kern"

    # Si no coincide con ningún caso, limpiamos el nombre base del fichero
    base_name = os.path.splitext(raw_name)[0]
    return base_name.replace('_', ' ').replace('-', ' ').strip()


def run_pipeline(pdf_filename: str, supabase_client: Client, embeddings_model: Embeddings, force_reparse: bool = False):
    """
    Orquesta el pipeline completo para un único archivo PDF:
    1. Convierte el PDF a Markdown.
    2. Crea chunks semánticos desde el Markdown.
    3. Genera embeddings y los ingesta en Supabase.
    """
    logging.info(f"--- Iniciando pipeline para el archivo: {pdf_filename} ---")
    
    # --- PASO 1: Conversión de PDF a Markdown (con caché) ---
    logging.info("Paso 1: Preparando conversión de PDF a Markdown...")
    pdf_file_path = os.path.join(config.DATA_PATH, pdf_filename)
    if not os.path.exists(pdf_file_path):
        logging.error(f"El archivo PDF '{pdf_file_path}' no se encontró. Abortando.")
        return

    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-')
    md_filename = f"parsed_by_{model_name_slug}_{pdf_filename.replace('.pdf', '.md')}"
    md_file_path = os.path.join(config.MARKDOWN_PATH, md_filename)

    # Lógica de caché: solo procesar si el archivo no existe o si se fuerza el re-parseo
    if not force_reparse and os.path.exists(md_file_path):
        logging.info(f"El archivo Markdown '{md_file_path}' ya existe. Saltando el paso de parsing de PDF.")
    else:
        logging.info("Iniciando conversión de PDF a Markdown...")
        # Asegurarse de que el directorio de salida para markdown exista
        if not os.path.exists(config.MARKDOWN_PATH):
            os.makedirs(config.MARKDOWN_PATH)

        generate_markdown_from_pdf_images(pdf_file_path, md_file_path)
        logging.info(f"PDF convertido a Markdown con éxito. Archivo guardado en: {md_file_path}")

    # --- PASO 2: Creación de Chunks ---
    logging.info("Paso 2: Creando chunks semánticos desde el archivo Markdown...")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Extraemos y luego estandarizamos el nombre del medicamento
    raw_medicine_name = extract_medicine_name(markdown_text, pdf_filename)
    medicine_name = standardize_medicine_name(raw_medicine_name)
    logging.info(f"Nombre del medicamento estandarizado a: '{medicine_name}'")

    semantic_blocks = markdown_to_semantic_blocks(markdown_text)
    
    # Cambiamos la función de chunking a nuestra nueva implementación de Sentence-Window
    chunks = create_sentence_window_chunks(
        blocks=semantic_blocks, 
        source_file=md_filename, 
        medicine_name=medicine_name
    )
    logging.info(f"Se crearon {len(chunks)} chunks (sentence-window) para este documento.")

    if not chunks:
        logging.warning("No se generaron chunks. Abortando el proceso de ingesta.")
        return

    # --- PASO 3: Generación de Embeddings ---
    logging.info("Paso 3: Generando embeddings con OpenAI...")
    texts_to_embed = [doc.page_content for doc in chunks]
    try:
        embeddings_list = embeddings_model.embed_documents(texts_to_embed)
        logging.info(f"Se generaron {len(embeddings_list)} embeddings.")
    except Exception as e:
        logging.error(f"Error al generar embeddings para {md_filename}: {e}", exc_info=True)
        return

    # --- PASO 4: Limpieza de Datos Antiguos en Supabase ---
    logging.info(f"Paso 4: Limpiando registros antiguos para '{md_filename}' en Supabase...")
    try:
        # Usamos el campo 'source' en los metadatos para identificar y borrar los chunks antiguos.
        # El operador '->>' extrae el valor del JSON como texto para una comparación exacta.
        supabase_client.table('documents').delete().eq('metadata->>source', md_filename).execute()
        logging.info("Limpieza de registros antiguos completada con éxito.")
    except Exception as e:
        # No detenemos el proceso si falla, ya que podría ser la primera vez que se ingesta.
        logging.warning(f"Advertencia durante la limpieza de datos antiguos: {e}")

    # --- PASO 5: Ingesta de Nuevos Datos en Supabase ---
    logging.info(f"Paso 5: Preparando y subiendo {len(chunks)} nuevos registros a Supabase...")
    records_to_insert = []
    for i, chunk in enumerate(chunks):
        records_to_insert.append({
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'embedding': embeddings_list[i]
        })

    try:
        supabase_client.table('documents').insert(records_to_insert).execute()
        logging.info(f"¡Éxito! Se han insertado {len(records_to_insert)} nuevos registros de {pdf_filename} en la base de datos.")
    except Exception as e:
        logging.error(f"Error al insertar datos para {pdf_filename}: {e}", exc_info=True)

    logging.info(f"--- Pipeline para {pdf_filename} finalizado. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Procesa e ingesta un prospecto de medicamento en formato PDF a la base de datos.")
    parser.add_argument(
        "pdf_filename", 
        type=str, 
        help="El nombre del archivo PDF a procesar (ej: 'nolotil_575.pdf'). Debe estar en la carpeta 'data'."
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Fuerza la re-conversión de PDF a Markdown aunque el archivo ya exista."
    )
    args = parser.parse_args()

    try:
        # --- Inicialización de Clientes ---
        logging.info("Inicializando y comprobando configuración...")
        config.check_env_vars() # Comprueba que las variables de entorno existan
        
        logging.info("Inicializando clientes de Supabase y Embeddings...")
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = get_embeddings_model()
        
        # --- Ejecución del Pipeline ---
        run_pipeline(args.pdf_filename, supabase, embeddings, args.force_reparse)

    except ValueError as e:
        logging.error(f"Error de configuración: {e}")
    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado: {e}", exc_info=True) 