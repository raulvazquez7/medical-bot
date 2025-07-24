import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markdown_it import MarkdownIt
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src import config
import re # Añadimos re para limpiar el nombre del medicamento

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

def create_chunks_from_blocks(blocks, source_file, chunk_size, chunk_overlap, medicine_name):
    """
    Toma los bloques semánticos, los divide en Documentos de LangChain,
    y antepone los metadatos (incluido el nombre del medicamento) a cada chunk
    para mantener el contexto.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = []
    
    for block in blocks:
        # 1. Separar el encabezado del resto del contenido.
        parts = block['content'].split('\n', 1)
        header = parts[0]
        content = parts[1].strip() if len(parts) > 1 else ""
        
        if not content:
            continue

        # 2. Dividir solo el contenido del bloque.
        split_texts = text_splitter.split_text(content)
        
        for text in split_texts:
            # 3. Construir los metadatos y el contenido final del chunk
            
            # Primero, creamos el diccionario de metadatos completo
            metadata = {
                "source": source_file,
                "path": block['path'],
                "medicine_name": medicine_name
            }
            
            # Ahora, creamos el texto que se va a "embeddear"
            final_content = f"""---
METADATOS:
- Medicamento: {metadata['medicine_name']}
- Fuente: {metadata['source']}
- Ruta: {metadata['path']}
---
CONTENIDO:
{header}

{text}
""".strip()
            
            all_chunks.append(Document(page_content=final_content, metadata=metadata))
            
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
        chunks = create_chunks_from_blocks(
            semantic_blocks, 
            source_file=INPUT_MD_NAME,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            medicine_name=test_medicine_name
        )
        print(f"Paso 2: Se han creado {len(chunks)} chunks en total.")
        
        if chunks:
            print("\nEjemplo de chunk generado:")
            print(chunks[0].page_content)
            print("\nMetadatos del chunk:")
            print(chunks[0].metadata)

        print("\nPrueba finalizada con éxito.")