import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == '__main__':
    DATA_PATH = 'data/'
    # Filtramos para procesar solo los archivos de paracetamol
    FILE_PREFIX_FILTER = 'paracetamol'
    
    # Instanciamos el splitter una sola vez con los parámetros definidos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=300)
    
    print("Iniciando el proceso de carga y chunking...\n")

    # Comprobar si la ruta de datos existe
    if not os.path.exists(DATA_PATH):
        print(f"Error: La ruta '{DATA_PATH}' no existe.")
    else:
        # Iteramos sobre cada archivo en el directorio de datos
        for file_name in os.listdir(DATA_PATH):
            # Aplicar filtro si se proporciona
            if FILE_PREFIX_FILTER and not file_name.lower().startswith(FILE_PREFIX_FILTER.lower()):
                continue
                
            if file_name.endswith('.pdf'):
                print(f"\n==================================================")
                print(f"Procesando archivo: {file_name}")
                print(f"==================================================")
                
                file_path = os.path.join(DATA_PATH, file_name)
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    
                    # Añadir metadatos de la fuente a cada página cargada
                    for doc in documents:
                        doc.metadata['source'] = file_name
                        
                    # Dividir solo el documento actual
                    chunks = text_splitter.split_documents(documents)
                    
                    print(f"--> Documento dividido en {len(chunks)} chunks.\n")
                    
                    # Imprimir cada chunk para análisis
                    for i, chunk in enumerate(chunks):
                        print(f"--- Chunk {i+1}/{len(chunks)} ---")
                        print(chunk.page_content)
                        print(f"Metadatos: {chunk.metadata}")
                        print("\n")
                        
                except Exception as e:
                    print(f"No se pudo cargar o procesar el archivo {file_name}: {e}")

    print("--- Proceso completado ---") 