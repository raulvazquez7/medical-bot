import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fitz  # PyMuPDF
import base64
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
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

# --- Definición del Esquema de Salida con Pydantic ---
class ProspectusMarkdown(BaseModel):
    """The complete prospectus converted to a clean and structured Markdown format."""
    markdown_content: str = Field(
        description="The full content of the document in Markdown format, following all formatting and structure rules specified in the prompt."
    )

def pdf_to_base64_images(pdf_path):
    """Convierte cada página de un PDF en una lista de imágenes en base64."""
    logging.info(f"Converting PDF '{pdf_path}' to images...")
    try:
        doc = fitz.open(pdf_path)
        base64_images = []
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(base64_image)
        doc.close()
        logging.info(f"PDF converted to {len(base64_images)} images.")
        return base64_images
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return []

def get_system_prompt_for_parsing():
    """Returns the detailed system prompt for the LLM."""
    return """
You are an expert in document processing (NLP) and structured text analysis. Your task is to analyze the IMAGES of a medication leaflet that I will provide and convert it to Markdown format. You must correctly identify and nest its hierarchical structure based on the visual layout of the text.

The goal is to preserve the relationships between sections, subsections, lists, and nested lists so that the resulting text is semantically coherent.

Expected Output Format:
- Main Sections: Level 1 Header (#).
- Subsections: Level 2 Header (##).
- List Contexts or Sub-subsections: Level 3 or 4 Header (### or ####).
- List Items: Dash (-).
- Nested List Items: Indented dash (    -).

Structure Rules to Follow:
1. Main Sections:
   They are easy to identify. They always start with a number, a period, and a space.
   Example in text: 1. What Enantyum is and what it is used for
   Markdown output: # 1. What Enantyum is and what it is used for

2. Subsections:
   These are the main headings within a section. They are not numbered and are usually bold in the original PDF.
   Example in text: Warnings and precautions
   Markdown output: ## Warnings and precautions

3. Lists and Hierarchies (The most important part):
   Lists can be simple or nested. The hierarchy is sometimes visual (with indentation) and sometimes implicit (contextual).

   A) Implicit Hierarchy (by Context):
   A paragraph can act as an introduction for a group of subheadings and their lists. You must recognize this dependency.
   Example in text:
   Always tell your doctor...
   Not recommended combinations:
   - Acetylsalicylic acid...
   Combinations requiring precautions:
   - ACE inhibitors...
   Logic to apply: The introductory paragraph ("Always tell your doctor...") should be a header (e.g., ###). The subheadings that follow are its children and should have a lower header level (e.g., ####).

   B) Explicit Hierarchy (by Indentation):
   This is the case for visually nested lists. You must use indentation to reflect the structure.
   Conceptual example:
   Tell your doctor...
   - Central nervous system depressants:
       - Major tranquilizers (antipsychotics).
   Logic to apply: The main list item (- Central nervous system depressants...) contains a secondary list, whose items (- Major tranquilizers...) must be indented in the Markdown.

Now, please process the images I will provide below and fill in the required data structure with the content converted to Markdown, faithfully following these rules and examples.
"""

def generate_markdown_from_pdf_images(pdf_path: str, output_path: str):
    """
    Main flow: converts a PDF to images, calls the LLM with them, and
    saves the result to a Markdown file.
    """
    base64_images = pdf_to_base64_images(pdf_path)
    if not base64_images:
        return

    logging.info(f"Initializing LLM model: {config.PDF_PARSE_MODEL}")
    llm = ChatGoogleGenerativeAI(model=config.PDF_PARSE_MODEL, temperature=0)

    structured_llm = llm.with_structured_output(ProspectusMarkdown)

    # Build the multimodal message with clear System and Human roles
    system_message = SystemMessage(content=get_system_prompt_for_parsing())
    
    human_message_content = []
    for img in base64_images:
        human_message_content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{img}"
        })
    
    human_message = HumanMessage(content=human_message_content)

    logging.info("Sending the request to the LLM. This may take several seconds...")
    try:
        # The prompt is now a list of structured messages
        prompt = [system_message, human_message]
        response = structured_llm.invoke(prompt)
        markdown_content = response.markdown_content
        
        logging.info("LLM returned a structured response successfully.")

        logging.info(f"Saving the result to '{output_path}'...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info("Process completed successfully!")

    except Exception as e:
        logging.error(f"Error calling the LLM or processing the response: {e}", exc_info=True)


if __name__ == '__main__':
    # --- Experiment Configuration ---
    # This block now serves as a quick test using the central configuration.
    # To process a new file, the main entry point is '03_ingest.py'.
    print("--- Running test for '01_pdf_to_markdown.py' ---")
    print("This script should not be run directly for ingestion. Use '03_ingest.py'.")
    
    # Use a test configuration
    INPUT_PDF_NAME = 'espidifen_600.pdf'
    
    # Build paths using the central configuration
    pdf_file_path = os.path.join(config.DATA_PATH, INPUT_PDF_NAME)
    
    # Create output filename
    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-') # For a cleaner filename
    md_file_name = f"parsed_by_{model_name_slug}_{INPUT_PDF_NAME.replace('.pdf', '.md')}"
    md_file_path = os.path.join(config.MARKDOWN_PATH, md_file_name)

    # Ensure the output directory exists
    if not os.path.exists(config.MARKDOWN_PATH):
        os.makedirs(config.MARKDOWN_PATH)

    if not os.path.exists(pdf_file_path):
        logging.error(f"Input file '{pdf_file_path}' not found.")
    else:
        generate_markdown_from_pdf_images(pdf_file_path, md_file_path)
        print(f"Test finished. Markdown has been saved to: {md_file_path}") 