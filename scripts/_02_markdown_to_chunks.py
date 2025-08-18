import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markdown_it import MarkdownIt
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from src import config
import re # Añadimos re para limpiar el nombre del medicamento

# --- Instalación única de NLTK (si es necesario) ---
# En algunos entornos, puede ser necesario descargar los datos de NLTK ('punkt').
# Descomenta las siguientes dos líneas y ejecuta este script directamente una vez si 
# obtienes un error relacionado con 'punkt' la primera vez que lo uses.
# import nltk
# nltk.download('punkt')

def markdown_to_semantic_blocks(markdown_text):
    """
    Parsea texto Markdown y lo agrupa en bloques semánticos basados en los encabezados.
    Un bloque contiene un encabezado y todo el contenido hasta el siguiente encabezado.
    """
    md = MarkdownIt()
    tokens = md.parse(markdown_text)
    
    blocks = []
    current_block_content = []
    current_path = []
    
    # Añadimos un token final "dummy" para asegurar que el último bloque se guarde
    tokens.append(type('Token', (), {'type': 'heading_open', 'tag': 'h0', 'content': 'End of Document', 'level': 0})())

    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            # Cuando encontramos un nuevo encabezado, guardamos el bloque anterior si tenía contenido.
            if current_block_content:
                blocks.append({
                    "path": " > ".join(current_path) if current_path else "General Information",
                    "content": "".join(current_block_content).strip()
                })
            
            level = int(token.tag[1]) if token.tag else 99
            
            header_content = ""
            if i + 1 < len(tokens) and tokens[i+1].type == 'inline':
                 header_content = tokens[i+1].content
            
            if level == 0:
                break
            
            while len(current_path) >= level:
                current_path.pop()
            current_path.append(header_content)
            
            # El contenido del bloque empieza con su encabezado
            current_block_content = [f"{'#' * level} {header_content}\n\n"]

        # Se ignora el párrafo si es parte de un item de lista para evitar duplicar texto
        elif token.type == 'paragraph_open':
            if i > 0 and tokens[i-1].type == 'list_item_open':
                continue
            content = tokens[i + 1].content if i + 1 < len(tokens) else ""
            current_block_content.append(f"{content}\n\n")

        elif token.type == 'bullet_list_open':
             current_block_content.append("\n")

        elif token.type == 'list_item_open':
            item_content = ""
            # Busca el contenido del item de lista en los siguientes tokens
            for next_token in tokens[i+1:]:
                if next_token.type == 'list_item_close' and next_token.level == token.level:
                    break
                if next_token.type == 'inline':
                    item_content = next_token.content
                    break
            
            indent = "    " * (token.level // 2)
            current_block_content.append(f"{indent}- {item_content}\n")

    return blocks

def create_sentence_window_chunks(blocks, source_file, medicine_name, window_size=2):
    """
    Toma bloques semánticos y crea chunks basados en oraciones individuales,
    añadiendo un "contexto de ventana" (N oraciones antes y después).
    Esta versión es más robusta ya que pre-procesa el Markdown para NLTK,
    manejando correctamente párrafos multi-línea y listas.
    """
    all_chunks = []
    sentence_splitter = NLTKTextSplitter(language='spanish')

    for block in blocks:
        sentences_in_block = []
        
        # Primero, eliminamos los encabezados del contenido para no procesarlos
        content_without_headers = "\n".join([line for line in block['content'].strip().split('\n') if not line.strip().startswith('#')])
        
        lines = content_without_headers.strip().split('\n')
        paragraph_buffer = []

        for line in lines:
            stripped_line = line.strip()
            # Si es un item de lista, es una unidad semántica por sí misma.
            if stripped_line.startswith('- '):
                # Primero, procesamos cualquier párrafo que estuviera en el buffer
                if paragraph_buffer:
                    full_paragraph = " ".join(paragraph_buffer)
                    sentences_in_block.extend(sentence_splitter.split_text(full_paragraph))
                    paragraph_buffer = [] # Limpiamos el buffer
                
                # Añadimos el item de la lista como su propia "oración"
                sentences_in_block.append(stripped_line.lstrip('- ').strip())
            # Si es una línea vacía, también indica un salto de párrafo
            elif not stripped_line:
                if paragraph_buffer:
                    full_paragraph = " ".join(paragraph_buffer)
                    sentences_in_block.extend(sentence_splitter.split_text(full_paragraph))
                    paragraph_buffer = []
            # Si no es un item de lista, lo añadimos al buffer del párrafo actual
            else:
                paragraph_buffer.append(stripped_line)

        # No olvidar procesar el último párrafo si el bloque no termina con una lista
        if paragraph_buffer:
            full_paragraph = " ".join(paragraph_buffer)
            sentences_in_block.extend(sentence_splitter.split_text(full_paragraph))

        # Filtramos cualquier resultado vacío que pueda haber quedado.
        sentences = [s for s in sentences_in_block if s]

        if not sentences:
            continue

        # 2. Iterar sobre cada oración para crear un chunk (esta lógica no cambia)
        for i, sentence in enumerate(sentences):
            start_index = max(0, i - window_size)
            end_index = min(len(sentences), i + window_size + 1)
            context_window = " ".join(sentences[start_index:end_index])
            
            metadata = {
                "source": source_file,
                "path": block['path'],
                "medicine_name": medicine_name,
                "main_sentence": sentence
            }
            
            final_content_to_embed = f"""---
METADATOS:
- Medicamento: {metadata['medicine_name']}
- Ruta: {metadata['path']}
---
CONTEXTO:
{context_window}
""".strip()
            
            all_chunks.append(Document(page_content=final_content_to_embed, metadata=metadata))
            
    return all_chunks


# Este bloque solo se ejecuta si corres "python scripts/_02_markdown_to_chunks.py" directamente.
# Sirve como una prueba rápida y limpia (smoke test).
if __name__ == '__main__':
    # --- Parámetros de prueba, locales a este bloque ---
    # Este bloque ahora sirve como una prueba rápida usando la configuración central.
    # Para procesar un nuevo archivo, el punto de entrada principal es '03_ingest.py'.
    print("--- Ejecutando prueba de '02_markdown_to_chunks.py' ---")
    print("Este script no debe ejecutarse directamente para la ingesta. Usar 'ingest.py'.")

    # Usamos un archivo de prueba que debería existir
    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-')
    INPUT_MD_NAME = f"parsed_by_{model_name_slug}_espidifen_600.md"
    
    md_file_path = os.path.join(config.MARKDOWN_PATH, INPUT_MD_NAME)

    if not os.path.exists(md_file_path):
        print(f"Error: El archivo Markdown de entrada '{md_file_path}' no se encontró.")
        print("Asegúrate de haber ejecutado primero la prueba de '01_pdf_to_markdown.py'")
    else:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        print("--- Ejecutando prueba de chunking ---")
        
        semantic_blocks = markdown_to_semantic_blocks(markdown_text)
        print(f"Paso 1: Se han creado {len(semantic_blocks)} bloques semánticos.")

        # Añadimos un nombre de medicamento de prueba
        test_medicine_name = "Medicamento de Prueba (Espidifen 600)"
        chunks = create_sentence_window_chunks(
            blocks=semantic_blocks, 
            source_file=INPUT_MD_NAME,
            medicine_name=test_medicine_name
        )
        print(f"Paso 2: Se han creado {len(chunks)} chunks en total.")
        
        if chunks:
            print("\nEjemplo de chunk generado:")
            print(chunks[0].page_content)
            print("\nMetadatos del chunk:")
            print(chunks[0].metadata)

        print("\nPrueba finalizada con éxito.")