import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fitz  # PyMuPDF
import base64
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage
from pydantic import BaseModel, Field
from src import config

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Imprime logs en la consola
    ]
)

# Cargar las variables de entorno (GOOGLE_API_KEY)
# load_dotenv() # Eliminado, ahora se gestiona en config.py

# --- Definición del Esquema de Salida con Pydantic ---
class ProspectusMarkdown(BaseModel):
    """El prospecto completo convertido a un formato Markdown limpio y estructurado."""
    markdown_content: str = Field(
        description="El contenido completo del documento en formato Markdown, siguiendo todas las reglas de formato y estructura especificadas en el prompt."
    )

def pdf_to_base64_images(pdf_path):
    """Convierte cada página de un PDF en una lista de imágenes en base64."""
    logging.info(f"Convirtiendo PDF '{pdf_path}' a imágenes...")
    try:
        doc = fitz.open(pdf_path)
        base64_images = []
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(base64_image)
        doc.close()
        logging.info(f"PDF convertido a {len(base64_images)} imágenes.")
        return base64_images
    except Exception as e:
        logging.error(f"Error al convertir el PDF a imágenes: {e}")
        return []

def get_llm_prompt():
    """Retorna el prompt detallado para el LLM, adaptado para recibir imágenes."""
    return """
Actúa como un experto en procesamiento de documentos (NLP) y análisis de texto estructurado. Tu tarea es analizar las IMÁGENES de un prospecto de medicamento que te proporcionaré y convertirlo a formato Markdown, identificando y anidando correctamente su estructura jerárquica basándote en la disposición visual del texto.

El objetivo es preservar las relaciones entre secciones, subsecciones, listas y listas anidadas para que el texto resultante sea semánticamente coherente.

Formato de Salida Esperado:
Secciones Principales: Encabezado de Nivel 1 (#).
Subsecciones: Encabezado de Nivel 2 (##).
Contextos de Lista o Sub-subsecciones: Encabezado de Nivel 3 o 4 (### o ####).
Ítems de Lista: Guion (-).
Ítems de Lista Anidada: Guion con indentación (    -).

Reglas de Estructura a Seguir:
1. Secciones Principales:
Son fáciles de identificar. Siempre empiezan con un número, un punto y un espacio.
Ejemplo en el texto: 1. Qué es Enantyum y para qué se utiliza
Salida en Markdown: # 1. Qué es Enantyum y para qué se utiliza

2. Subsecciones:
Son los títulos principales dentro de una sección. No están numerados y en el PDF original suelen estar en negrita.
Ejemplo en el texto: Advertencias y precauciones
Salida en Markdown: ## Advertencias y precauciones

3. Listas y Jerarquías (La parte más importante):
Las listas pueden ser simples o anidadas. La jerarquía a veces es visual (con sangría) y otras veces es implícita (contextual).

A) Jerarquía Implícita (por Contexto):
Un párrafo puede actuar como una introducción para un grupo de sub-títulos y sus listas. Debes reconocer esta dependencia.
Ejemplo en el texto:
Informe siempre a su médico...
Asociaciones no recomendadas:
- Ácido acetilsalicílico...
Asociaciones que requieren precaución:
- Inhibidores de la ECA...
Lógica a aplicar: El párrafo introductorio ("Informe siempre...") debe ser un encabezado (ej. ###). Los subtítulos que le siguen son sus hijos y deben tener un nivel de encabezado inferior (ej. ####).

B) Jerarquía Explícita (por Indentación):
Este es el caso de las listas anidadas visualmente. Debes usar la indentación para reflejar la estructura.
Ejemplo conceptual:
Comunique a su médico...
- Depresores del sistema nervioso central:
    - Tranquilizantes mayores (antipsicóticos).
Lógica a aplicar: El ítem de la lista principal (- Depresores...) contiene una lista secundaria, cuyos ítems (- Tranquilizantes...) deben ir indentados en el Markdown.

Ahora, por favor, procesa las imágenes que te proporcionaré a continuación y rellena la estructura de datos requerida con el contenido convertido a formato Markdown, siguiendo fielmente estas reglas y ejemplos.
"""

def generate_markdown_from_pdf_images(pdf_path: str, output_path: str):
    """
    Flujo principal: convierte un PDF a imágenes, llama al LLM con ellas y
    guarda el resultado en un archivo Markdown.
    """
    base64_images = pdf_to_base64_images(pdf_path)
    if not base64_images:
        return

    logging.info(f"Inicializando el modelo LLM: {config.PDF_PARSE_MODEL}")
    llm = ChatGoogleGenerativeAI(model=config.PDF_PARSE_MODEL, temperature=0)

    # El método recomendado y más directo de LangChain para forzar la salida estructurada.
    structured_llm = llm.with_structured_output(ProspectusMarkdown)

    # Construimos el mensaje multimodal
    prompt_parts = [
        {"type": "text", "text": get_llm_prompt()},
    ]
    for img in base64_images:
        prompt_parts.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{img}"
        })

    message = HumanMessage(content=prompt_parts)

    logging.info("Enviando la petición al LLM. Esto puede tardar varios segundos...")
    try:
        # La respuesta será directamente un objeto Pydantic, no un AIMessage
        response = structured_llm.invoke([message])
        markdown_content = response.markdown_content
        
        logging.info("El LLM ha devuelto una respuesta estructurada con éxito.")

        logging.info(f"Guardando el resultado en '{output_path}'...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info("¡Proceso completado con éxito!")

    except Exception as e:
        logging.error(f"Error al llamar al LLM o procesar la respuesta: {e}", exc_info=True)


if __name__ == '__main__':
    # --- Configuración del Experimento ---
    # Este bloque ahora sirve como una prueba rápida usando la configuración central.
    # Para procesar un nuevo archivo, el punto de entrada principal es '03_ingest.py'.
    print("--- Ejecutando prueba de '01_pdf_to_markdown.py' ---")
    print("Este script no debe ejecutarse directamente para la ingesta. Usar '03_ingest.py'.")
    
    # Usamos una configuración de prueba
    INPUT_PDF_NAME = 'espidifen_600.pdf'
    
    # Construir rutas usando la configuración central
    pdf_file_path = os.path.join(config.DATA_PATH, INPUT_PDF_NAME)
    
    # Crear nombre de archivo de salida
    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-') # Para un nombre de archivo más limpio
    md_file_name = f"parsed_by_{model_name_slug}_{INPUT_PDF_NAME.replace('.pdf', '.md')}"
    md_file_path = os.path.join(config.MARKDOWN_PATH, md_file_name)

    # Asegurarse de que el directorio de salida exista
    if not os.path.exists(config.MARKDOWN_PATH):
        os.makedirs(config.MARKDOWN_PATH)

    if not os.path.exists(pdf_file_path):
        logging.error(f"El archivo de entrada '{pdf_file_path}' no se encontró.")
    else:
        generate_markdown_from_pdf_images(pdf_file_path, md_file_path)
        print(f"Prueba finalizada. El Markdown se ha guardado en: {md_file_path}") 