import os
import sys
# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import re
from docling.document_converter import DocumentConverter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src import config

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def clean_markdown_artifacts(raw_markdown: str) -> str:
    """
    Realiza una limpieza programática del Markdown generado por Docling,
    eliminando artefactos y normalizando el formato.
    """
    logging.info("Limpiando artefactos del Markdown generado...")
    
    # 1. Eliminar placeholders de imagen
    cleaned_markdown = re.sub(r'^\s*<!-- image -->\s*\n?', '', raw_markdown, flags=re.MULTILINE)
    
    # 2. Normalizar marcadores de lista (ej: '- ' y '- -')
    cleaned_markdown = re.sub(r'^- \s*', '- ', cleaned_markdown, flags=re.MULTILINE)
    cleaned_markdown = re.sub(r'^- -\s*', '- ', cleaned_markdown, flags=re.MULTILINE)
    
    # 3. Consolidar múltiples líneas en blanco a una sola
    cleaned_markdown = re.sub(r'\n{3,}', '\n\n', cleaned_markdown)
    
    logging.info("Limpieza de artefactos completada.")
    return cleaned_markdown.strip()

def refine_hierarchy_with_llm(markdown_text: str) -> str:
    """
    Utiliza un LLM para analizar el Markdown y corregir la jerarquía de
    los encabezados (#, ##, ###) para que refleje una anidación lógica.
    """
    logging.info(f"Usando el modelo '{config.PDF_PARSE_MODEL}' para refinar la jerarquía de encabezados...")
    
    try:
        llm = ChatGoogleGenerativeAI(model=config.PDF_PARSE_MODEL, temperature=0.0)
        
        system_prompt = """
Eres un asistente experto en la re-estructuración de documentos Markdown.
Tu única tarea es analizar el Markdown de un prospecto médico y corregir la jerarquía de los encabezados (`#`, `##`, `###`, etc.) para que refleje una anidación lógica.

Reglas que debes seguir ESTRICTAMENTE:
1.  NO alteres, añadas ni elimines NINGUNA palabra del contenido del texto.
2.  Tu única modificación permitida es cambiar el número de almohadillas (#) al principio de las líneas que son encabezados.
3.  Los títulos principales (ej: "1. Qué es Nolotil...") deben ser de nivel 1 (`#`).
4.  Las secciones dentro de un título principal (ej: "Advertencias y precauciones") deben ser de nivel 2 (`##`).
5.  Las subsecciones dentro de una sección (ej: "Problemas hepáticos" dentro de "Advertencias") deben ser de nivel 3 (`###`).
6.  Devuelve el documento Markdown COMPLETO con la jerarquía corregida.
"""
        
        human_prompt = "Por favor, corrige la jerarquía del siguiente Markdown:\n\n{markdown_content}"
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        
        chain = chat_prompt | llm
        
        response = chain.invoke({"markdown_content": markdown_text})
        
        logging.info("La jerarquía ha sido refinada por el LLM con éxito.")
        return response.content
        
    except Exception as e:
        logging.error(f"Error durante el refinamiento de jerarquía con el LLM: {e}", exc_info=True)
        # En caso de error, devolvemos el texto original para no perder el trabajo previo
        return markdown_text

def parse_pdf_with_docling(pdf_path: str, output_path: str, force_reparse: bool = False):
    """
    Orquesta el pipeline híbrido: Docling para parsing, Python para limpieza y LLM para jerarquía.
    """
    logging.info(f"--- Iniciando pipeline híbrido para: {os.path.basename(pdf_path)} ---")
    
    if not force_reparse and os.path.exists(output_path):
        logging.info(f"El archivo '{output_path}' ya existe. Saltando. Usa --force-reparse para sobreescribir.")
        return

    if not os.path.exists(pdf_path):
        logging.error(f"El archivo de entrada no se encontró: {pdf_path}")
        return

    try:
        # PASO 1: Parsing rápido con Docling
        logging.info("Paso 1: Inicializando el convertidor de Docling y procesando el PDF...")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        if not hasattr(result, "document") or result.document is None:
            logging.error("Docling no devolvió un documento procesado.")
            return

        if hasattr(result.document, "export_to_markdown"):
            raw_markdown_content = result.document.export_to_markdown()
        else:
            raise RuntimeError("No se encontró 'export_to_markdown'. Revisa la versión de Docling.")
        
        # PASO 2: Limpieza programática de artefactos
        cleaned_markdown = clean_markdown_artifacts(raw_markdown_content)
        
        # PASO 3: Refinamiento de jerarquía con LLM
        final_markdown = refine_hierarchy_with_llm(cleaned_markdown)
        
        # PASO 4: Guardar el resultado final
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        
        logging.info(f"¡Éxito! El resultado final ha sido guardado en: {output_path}")

    except Exception as e:
        logging.error(f"Ha ocurrido un error durante el pipeline de Docling: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Script híbrido para convertir un PDF a MD: Docling (parsing) + LLM (jerarquía)."""
    )
    parser.add_argument(
        "pdf_filename", 
        type=str, 
        help="El nombre del archivo PDF a procesar (ej: 'nolotil_575.pdf'). Debe estar en 'data'."
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Fuerza la re-conversión del PDF aunque el archivo Markdown ya exista."
    )
    args = parser.parse_args()

    # --- Construcción de Rutas ---
    input_pdf_path = os.path.join(config.DATA_PATH, args.pdf_filename)
    
    # Hemos cambiado el nombre del modelo que usamos, así que actualizamos el slug
    model_name_slug = f"docling_then_{config.PDF_PARSE_MODEL.replace('.', '-')}"
    output_md_filename = f"parsed_by_{model_name_slug}_{args.pdf_filename.replace('.pdf', '.md')}"
    output_md_path = os.path.join(config.MARKDOWN_PATH, output_md_filename)

    if not os.path.exists(config.MARKDOWN_PATH):
        os.makedirs(config.MARKDOWN_PATH)

    parse_pdf_with_docling(input_pdf_path, output_md_path, args.force_reparse)
