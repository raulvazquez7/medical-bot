import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings

# Importamos la configuración central y las funciones de los otros scripts
from src import config
from scripts._01_pdf_to_markdown import generate_markdown_from_pdf_images
from scripts._02_markdown_to_chunks import markdown_to_semantic_blocks, create_chunks_from_blocks


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
            print(f"Nombre del medicamento extraído: '{medicine_name}'")
            return medicine_name
    except Exception:
        pass  # Si hay cualquier error en la regex, usamos el fallback

    # Si no se encuentra el patrón, usamos el nombre del archivo como fallback
    name_from_file = os.path.splitext(fallback_filename)[0].replace('_', ' ').replace('-', ' ')
    print(f"No se pudo extraer el nombre del prospecto. Usando fallback: '{name_from_file}'")
    return name_from_file


def run_pipeline(pdf_filename: str, supabase_client: Client, embeddings_model: OpenAIEmbeddings):
    """
    Orquesta el pipeline completo para un único archivo PDF:
    1. Convierte el PDF a Markdown.
    2. Crea chunks semánticos desde el Markdown.
    3. Genera embeddings y los ingesta en Supabase.
    """
    print(f"\n--- Iniciando pipeline para el archivo: {pdf_filename} ---")
    
    # --- PASO 1: Conversión de PDF a Markdown ---
    print("Paso 1: Convirtiendo PDF a Markdown...")
    pdf_file_path = os.path.join(config.DATA_PATH, pdf_filename)
    if not os.path.exists(pdf_file_path):
        print(f"Error: El archivo PDF '{pdf_file_path}' no se encontró. Abortando.")
        return

    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-')
    md_filename = f"parsed_by_{model_name_slug}_{pdf_filename.replace('.pdf', '.md')}"
    md_file_path = os.path.join(config.MARKDOWN_PATH, md_filename)

    # Asegurarse de que el directorio de salida para markdown exista
    if not os.path.exists(config.MARKDOWN_PATH):
        os.makedirs(config.MARKDOWN_PATH)

    generate_markdown_from_pdf_images(pdf_file_path, md_file_path)
    print(f"PDF convertido a Markdown con éxito. Archivo guardado en: {md_file_path}")

    # --- PASO 2: Creación de Chunks ---
    print("\nPaso 2: Creando chunks semánticos desde el archivo Markdown...")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Extraemos el nombre del medicamento ANTES de hacer el chunking
    medicine_name = extract_medicine_name(markdown_text, pdf_filename)

    semantic_blocks = markdown_to_semantic_blocks(markdown_text)
    chunks = create_chunks_from_blocks(
        semantic_blocks, 
        source_file=md_filename, 
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP,
        medicine_name=medicine_name  # <-- Pasamos el nombre extraído
    )
    print(f"Se crearon {len(chunks)} chunks para este documento.")

    if not chunks:
        print("No se generaron chunks. Abortando el proceso de ingesta.")
        return

    # --- PASO 3: Generación de Embeddings ---
    print("\nPaso 3: Generando embeddings con OpenAI...")
    texts_to_embed = [doc.page_content for doc in chunks]
    try:
        embeddings_list = embeddings_model.embed_documents(texts_to_embed)
        print(f"Se generaron {len(embeddings_list)} embeddings.")
    except Exception as e:
        print(f"Error al generar embeddings para {md_filename}: {e}")
        return

    # --- PASO 4: Limpieza de Datos Antiguos en Supabase ---
    print(f"\nPaso 4: Limpiando registros antiguos para '{md_filename}' en Supabase...")
    try:
        # Usamos el campo 'source' en los metadatos para identificar y borrar los chunks antiguos.
        # El operador '->>' extrae el valor del JSON como texto para una comparación exacta.
        supabase_client.table('documents').delete().eq('metadata->>source', md_filename).execute()
        print("Limpieza de registros antiguos completada con éxito.")
    except Exception as e:
        # No detenemos el proceso si falla, ya que podría ser la primera vez que se ingesta.
        print(f"Advertencia durante la limpieza de datos antiguos: {e}")

    # --- PASO 5: Ingesta de Nuevos Datos en Supabase ---
    print(f"\nPaso 5: Preparando y subiendo {len(chunks)} nuevos registros a Supabase...")
    records_to_insert = []
    for i, chunk in enumerate(chunks):
        records_to_insert.append({
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'embedding': embeddings_list[i]
        })

    try:
        supabase_client.table('documents').insert(records_to_insert).execute()
        print(f"¡Éxito! Se han insertado {len(records_to_insert)} nuevos registros de {pdf_filename} en la base de datos.")
    except Exception as e:
        print(f"Error al insertar datos para {pdf_filename}: {e}")

    print(f"\n--- Pipeline para {pdf_filename} finalizado. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Procesa e ingesta un prospecto de medicamento en formato PDF a la base de datos.")
    parser.add_argument(
        "pdf_filename", 
        type=str, 
        help="El nombre del archivo PDF a procesar (ej: 'nolotil_575.pdf'). Debe estar en la carpeta 'data'."
    )
    args = parser.parse_args()

    try:
        # --- Inicialización de Clientes ---
        print("Inicializando y comprobando configuración...")
        config.check_env_vars() # Comprueba que las variables de entorno existan
        
        print("Inicializando clientes de Supabase y OpenAI...")
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDINGS_MODEL)
        
        # --- Ejecución del Pipeline ---
        run_pipeline(args.pdf_filename, supabase, embeddings)

    except ValueError as e:
        print(f"Error de configuración: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}") 