import fitz  # PyMuPDF
import os
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Parámetros de configuración ---
CHUNK_SIZE = 1700
CHUNK_OVERLAP = 300
X_TOLERANCE = 70 
MAX_LEN_TITLE = 150

def _extract_sections(doc):
    """
    Función auxiliar para extraer los títulos de sección y subsección de un documento.
    (Basada en la lógica de test_bold_extraction.py)
    """
    sections = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
        for b in blocks:
            if b.get('type') == 0:
                for l in b.get("lines", []):
                    if not l.get("spans"):
                        continue
                        
                    line_text = "".join(s["text"] for s in l["spans"]).strip()
                    if not line_text:
                        continue

                    starts_with_uppercase = not line_text[0].islower()
                    first_span = l["spans"][0]
                    
                    is_bold_start = "bold" in first_span["font"].lower()
                    is_left_aligned = l['bbox'][0] < X_TOLERANCE
                    is_short_enough = len(line_text) < MAX_LEN_TITLE
                    
                    if is_bold_start and is_left_aligned and is_short_enough and starts_with_uppercase:
                        sections.append({
                            "text": line_text,
                            "page": page_num + 1,
                            "bbox": l['bbox'] # Guardamos el Bounding Box para futuras referencias
                        })
    return sections

def _extract_medication_name(doc, sections):
    """
    Extrae el nombre del medicamento, que suele estar en la primera página,
    antes de la primera sección principal.
    """
    # Límite superior: el inicio de la primera sección.
    first_section_y = sections[0]['bbox'][1] if sections else 700

    # Límite inferior: la parte superior de la página
    y_start_limit = 100 # Ignorar cabeceras de página muy altas

    # Buscamos el texto con el tamaño de fuente más grande en el área de interés
    max_font_size = 0
    candidate_lines = []

    page = doc[0] # Solo buscamos en la primera página
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if b.get('type') == 0:
            for l in b.get("lines", []):
                # Comprobamos si la línea está en el área de interés vertical
                if y_start_limit < l['bbox'][1] < first_section_y:
                    for s in l["spans"]:
                         # Descartamos texto genérico que no es el nombre
                        if "prospecto" in s["text"].lower() or "información" in s["text"].lower():
                            continue
                        candidate_lines.append(s)
                        if s["size"] > max_font_size:
                            max_font_size = s["size"]
    
    # Filtramos los candidatos finales por un tamaño de fuente cercano al máximo
    # para capturar todas las dosis (ej. Eutirox 50mg, Eutirox 75mg, etc.)
    final_names = [
        line["text"].strip() for line in candidate_lines 
        if line["size"] > max_font_size * 0.9
    ]

    # Unimos los nombres en caso de que haya varios, como en Eutirox
    return ", ".join(dict.fromkeys(final_names)) # `dict.fromkeys` para eliminar duplicados

def extract_pdf_content(pdf_path):
    """
    Función principal que orquesta la extracción de todo el contenido
    relevante de un prospecto en PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"error": f"No se pudo abrir el archivo {pdf_path}: {e}"}

    # 1. Extraer las secciones/subsecciones
    sections = _extract_sections(doc)
    if not sections:
        return {"error": f"No se encontraron secciones en {pdf_path}."}

    # 2. Extraer el nombre del medicamento
    medication_name = _extract_medication_name(doc, sections)

    # 3. CORRECCIÓN: Extraer el texto completo del documento página por página
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    doc.close()

    return {
        "medication_name": medication_name,
        "sections": sections,
        "full_text": full_text
    }

# --- NUEVA FUNCIÓN DE CHUNKING ---
def chunk_document_by_structure(file_name, medication_name, sections, full_text):
    """
    Divide el texto del documento en chunks basados en la estructura de secciones
    y subsecciones, asignando metadatos jerárquicos.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    current_main_section = "Información General" # Default

    for i, section in enumerate(sections):
        section_title = section["text"]
        
        # Identificar la sección principal actual
        if re.match(r"^\d+\.\s", section_title):
            current_main_section = section_title

        # Determinar el inicio y el fin del contenido de esta sección
        start_pos = full_text.find(section_title)
        if start_pos == -1:
            continue
        
        end_pos = len(full_text)
        if i + 1 < len(sections):
            next_section_title = sections[i+1]["text"]
            next_pos = full_text.find(next_section_title, start_pos)
            if next_pos != -1:
                end_pos = next_pos
        
        # El contenido del bloque es el título + el texto que le sigue
        block_content = full_text[start_pos:end_pos]
        
        # Dividir el bloque si es largo, o tratarlo como un único chunk si es corto
        split_texts = text_splitter.split_text(block_content)
        
        for text in split_texts:
            metadata = {
                "source": file_name,
                "medication_name": medication_name,
                "main_section": current_main_section,
                "subsection": section_title,
                "page": section["page"]
            }
            chunks.append(Document(page_content=text, metadata=metadata))

    return chunks

if __name__ == '__main__':
    DATA_PATH = 'data/'
    # Prueba con Eutirox, que tiene varios nombres de medicamento
    test_file_name = 'eutirox.pdf' 
    test_file_path = os.path.join(DATA_PATH, test_file_name)

    if not os.path.exists(test_file_path):
        print(f"El archivo de prueba '{test_file_name}' no se encontró.")
    else:
        print(f"--- PASO 1: Extrayendo contenido de: {test_file_name} ---")
        content = extract_pdf_content(test_file_path)

        if "error" in content:
            print(content["error"])
        else:
            print("Extracción completada.")
            
            print(f"\n--- PASO 2: Creando chunks para: {test_file_name} ---")
            chunks = chunk_document_by_structure(
                file_name=test_file_name,
                medication_name=content['medication_name'],
                sections=content['sections'],
                full_text=content['full_text']
            )
            
            if chunks:
                print(f"¡Chunking completado! Se han creado {len(chunks)} chunks.")
                
                print("\n--- Inspección de TODOS los Chunks ---")
                for i, chunk in enumerate(chunks):
                    print(f"\n==================== CHUNK {i+1}/{len(chunks)} ====================")
                    print(f"METADATOS: {chunk.metadata}")
                    print("-------------------- TEXTO --------------------")
                    print(chunk.page_content)
            else:
                print("No se pudieron crear chunks para el documento.") 