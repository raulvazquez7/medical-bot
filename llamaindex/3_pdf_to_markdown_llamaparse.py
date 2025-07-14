import os
import glob
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse

# LlamaParse utiliza async, esto es necesario para ejecutarlo en un script síncrono
nest_asyncio.apply()

# Cargar las variables de entorno (LLAMA_CLOUD_API_KEY)
load_dotenv()

def run_parsing_experiment(pdf_files_to_process, parser_config):
    """
    Ejecuta un experimento de parsing sobre una lista de archivos PDF con una configuración
    específica de LlamaParse, procesando todas las páginas y extrayendo metadatos.
    """
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("Error: La variable de entorno LLAMA_CLOUD_API_KEY no está configurada.")
        return

    print("1. Inicializando LlamaParse con la siguiente configuración:")
    print(parser_config)
    parser = LlamaParse(**parser_config)

    for pdf_path in pdf_files_to_process:
        file_name = os.path.basename(pdf_path)
        print(f"\n--- Procesando archivo: {file_name} ---")
        
        md_file_path = os.path.join(
            "data_markdown", 
            f"parsed_{file_name.replace('.pdf', '.md')}"
        )
        
        try:
            # .load_data() es una corutina, por eso usamos nest_asyncio
            documents = parser.load_data(pdf_path)
            
            if documents:
                # --- Extracción de metadatos y logging ---
                num_docs = len(documents)
                processed_pages = set()
                # LlamaParse puede devolver un 'Document' por página. Recopilamos las páginas.
                for doc in documents:
                    if 'page_label' in doc.metadata:
                        processed_pages.add(doc.metadata['page_label'])
                
                num_pages_processed = len(processed_pages)

                print(f"📄 Metadatos para {file_name}:")
                print(f"   - Documentos devueltos por el parser: {num_docs}")
                print(f"   - Páginas procesadas: {num_pages_processed}")
                print("   - Consumo de créditos: La API no devuelve este dato. Por favor, consúltalo en tu dashboard de LlamaCloud.")

                # --- Concatenación y guardado del contenido completo ---
                # Unimos el texto de todos los documentos devueltos
                full_markdown_content = "\n\n---\n\n".join([doc.text for doc in documents])
                
                with open(md_file_path, "w", encoding="utf-8") as f:
                    f.write(full_markdown_content)
                print(f"✅ Éxito! Contenido de todas las páginas guardado en '{md_file_path}'")
            else:
                print(f"⚠️  LlamaParse no devolvió contenido para {file_name}.")

        except Exception as e:
            print(f"❌ Error procesando {file_name}: {e}")

if __name__ == '__main__':
    # --- Configuración del Experimento ---

    # 1. DEFINE AQUÍ LOS ARCHIVOS QUE QUIERES PROBAR
    PDF_FILES = [
        'enantyium_25_comprimidos.pdf'
    ]
    # Para procesar todos los PDFs de la carpeta data:
    # PDF_FILES = glob.glob('data/*.pdf') 

    # 2. DEFINE AQUÍ LA CONFIGURACIÓN DE LLAMAPARSE QUE QUIERES PROBAR
    PARSER_CONFIG = {
        "result_type": "markdown",
        "verbose": True,
        "language": "es",
        "num_workers": 2
    }

    # --- Ejecución del Experimento ---
    MARKDOWN_OUTPUT_PATH = 'data_markdown'
    if not os.path.exists(MARKDOWN_OUTPUT_PATH):
        os.makedirs(MARKDOWN_OUTPUT_PATH)

    # Convertimos los nombres de archivo a rutas completas
    files_to_process = []
    if isinstance(PDF_FILES, list):
        # Si es una lista de nombres
        files_to_process = [os.path.join('data', f) for f in PDF_FILES if os.path.exists(os.path.join('data', f))]
    elif isinstance(PDF_FILES, str) and '*' in PDF_FILES:
        # Si es un patrón glob
        files_to_process = glob.glob(PDF_FILES)

    if not files_to_process:
        print("No se encontraron archivos PDF para procesar. Revisa la configuración de PDF_FILES.")
    else:
        run_parsing_experiment(files_to_process, PARSER_CONFIG) 