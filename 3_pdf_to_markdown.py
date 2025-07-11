import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage

# Cargar las variables de entorno (GOOGLE_API_KEY)
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extrae el texto completo de un archivo PDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error al leer el PDF {pdf_path}: {e}")
        return None

def get_llm_prompt():
    """Retorna el prompt detallado para el LLM."""
    # Este es el prompt que has diseñado
    return """
Actúa como un experto en procesamiento de documentos (NLP) y análisis de texto estructurado. Tu tarea es analizar el texto de un prospecto de medicamento que te proporcionaré y convertirlo a formato Markdown, identificando y anidando correctamente su estructura jerárquica.

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

Ahora, por favor, procesa el texto completo que te proporcionaré a continuación y genera la salida en formato Markdown siguiendo fielmente estas reglas y ejemplos.
"""

def generate_markdown_from_pdf(pdf_path, output_path):
    """
    Flujo principal: lee un PDF, extrae su texto, llama al LLM y guarda el
    resultado en un archivo Markdown.
    """
    print(f"1. Extrayendo texto de '{pdf_path}'...")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return

    print("2. Inicializando el modelo LLM (Gemini 1.5 Flash)...")
    # Usamos Gemini 1.5 Flash, que es rápido, tiene una gran ventana de contexto y es muy capaz.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    system_prompt = get_llm_prompt()
    human_prompt = f"Aquí está el texto del prospecto:\n\n---\n\n{pdf_text}"
    
    print("3. Enviando la petición al LLM. Esto puede tardar unos segundos...")
    try:
        response = llm.invoke([
            HumanMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        markdown_content = response.content
        
        print(f"4. Guardando el resultado en '{output_path}'...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print("¡Proceso completado con éxito!")

    except Exception as e:
        print(f"Error al llamar al LLM: {e}")


if __name__ == '__main__':
    # --- Configuración ---
    DATA_PATH = 'data/'
    # Usamos el PDF de Enantyum como en tu prompt de ejemplo
    INPUT_PDF_NAME = 'enantyium_25_comprimidos.pdf'
    
    # Creamos una carpeta para guardar los Markdowns generados
    MARKDOWN_OUTPUT_PATH = 'data_markdown'
    if not os.path.exists(MARKDOWN_OUTPUT_PATH):
        os.makedirs(MARKDOWN_OUTPUT_PATH)

    pdf_file_path = os.path.join(DATA_PATH, INPUT_PDF_NAME)
    md_file_path = os.path.join(MARKDOWN_OUTPUT_PATH, INPUT_PDF_NAME.replace('.pdf', '.md'))

    if not os.path.exists(pdf_file_path):
        print(f"Error: El archivo de entrada '{pdf_file_path}' no se encontró.")
    else:
        generate_markdown_from_pdf(pdf_file_path, md_file_path) 