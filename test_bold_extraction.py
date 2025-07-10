import fitz  # PyMuPDF
import os

# --- Parámetros de configuración para los filtros ---
X_TOLERANCE = 70 
MAX_LEN = 150

def extract_sections_from_pdf(pdf_path):
    """
    Abre un archivo PDF e identifica los títulos de sección/subsección
    basándose en el formato (negrita) y la estructura (posición, mayúsculas, etc.).
    """
    sections = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error al abrir el archivo {pdf_path}: {e}")
        return []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
        for b in blocks:
            if b.get('type') == 0: # Procesar solo bloques de texto
                for l in b.get("lines", []):
                    if not l.get("spans"):
                        continue
                        
                    line_text = "".join(s["text"] for s in l["spans"]).strip()
                    if not line_text:
                        continue

                    # --- NUEVA REGLA: El título no debe empezar en minúscula ---
                    starts_with_uppercase = not line_text[0].islower()
                    
                    first_span = l["spans"][0]
                    
                    is_bold_start = "bold" in first_span["font"].lower()
                    is_left_aligned = l['bbox'][0] < X_TOLERANCE
                    is_short_enough = len(line_text) < MAX_LEN
                    
                    # Añadimos la nueva condición al filtro principal
                    if is_bold_start and is_left_aligned and is_short_enough and starts_with_uppercase:
                        sections.append({
                            "text": line_text,
                            "page": page_num + 1
                        })
                        
    return sections

if __name__ == '__main__':
    DATA_PATH = 'data/'
    # Nos centramos en el archivo que da problemas
    test_file_name = 'eutirox.pdf' 
    test_file_path = os.path.join(DATA_PATH, test_file_name)

    if not os.path.exists(test_file_path):
        print(f"El archivo de prueba '{test_file_name}' no se encontró en la carpeta '{DATA_PATH}'.")
    else:
        print(f"Analizando el archivo: {test_file_name} para encontrar secciones y subsecciones...")
        extracted_sections = extract_sections_from_pdf(test_file_path)

        if extracted_sections:
            print("\n--- Secciones/Subsecciones encontradas (filtradas): ---")
            for sec in extracted_sections:
                print(f"Página {sec['page']}: {sec['text']}")
        else:
            print("\nNo se encontraron secciones con los filtros actuales.") 