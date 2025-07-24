import sys
import os
# Añadimos la ruta raíz para que Python pueda encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from typing import List
from collections import defaultdict
from supabase import Client, create_client
from langchain.schema import Document, BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src import config

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- El Retriever Personalizado para Supabase ---
class SupabaseRetriever(BaseRetriever):
    """
    Un retriever personalizado que busca documentos en Supabase
    usando la función `match_documents` que creamos.
    """
    supabase_client: Client
    embeddings_model: OpenAIEmbeddings
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Dada una consulta de usuario, la convierte en un embedding y busca
        en la base de datos los chunks más relevantes.
        """
        logging.info(f"Generando embedding para la consulta: '{query}'")
        query_embedding = self.embeddings_model.embed_query(query)
        
        logging.info(f"Buscando los {self.top_k} documentos más relevantes en Supabase...")
        
        response = self.supabase_client.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': self.top_k
            }
        ).execute()

        if response.data:
            logging.info(f"Se encontraron {len(response.data)} documentos.")
            return [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in response.data]
        
        logging.warning("No se encontraron documentos relevantes.")
        return []

# --- Plantilla de Prompt para el RAG ---
RAG_PROMPT_TEMPLATE = """
Eres un asistente experto en farmacología y tu única función es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado de un prospecto de medicamento. Eres preciso, riguroso y nunca inventas información.

CONTEXTO:
---------------------
{context}
---------------------

PREGUNTA: {question}

RESPUESTA (clara y concisa):
"""

def format_docs(docs: List[Document]) -> str:
    """Función auxiliar para formatear los documentos recuperados en un único string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def run_chatbot():
    """Inicializa y ejecuta el bucle principal del chatbot de consola."""
    try:
        logging.info("Inicializando y comprobando configuración...")
        config.check_env_vars()
        
        logging.info("Inicializando clientes...")
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDINGS_MODEL)
        
        logging.info(f"Configurando el modelo de chat: {config.CHAT_MODEL_TO_USE}")
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        elif "gpt" in config.CHAT_MODEL_TO_USE:
            llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            raise ValueError(f"Modelo de chat no soportado: {config.CHAT_MODEL_TO_USE}. Revisa la configuración en 'src/config.py'.")

        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # --- Cadena RAG con Fuentes ---
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_sources = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        print("\n--- Medical Bot Iniciado ---")
        print("¡Hola! Estoy listo para responder tus preguntas sobre medicamentos.")
        print("Escribe 'salir' para terminar la conversación.")
        
        while True:
            question = input("\nTu pregunta: ")
            if question.lower() in ['salir', 'exit', 'quit']:
                print("Hasta luego. ¡Cuídate!")
                break
            
            print("\nBuscando en la base de datos y generando respuesta...")
            result = rag_chain_with_sources.invoke(question)
            
            print("\nRespuesta del Bot:")
            print(result["answer"])
            
            print("\n--- Fuentes Consultadas ---")
            if result["context"]:
                sources_by_doc = defaultdict(list)
                for doc in result["context"]:
                    source_file = doc.metadata.get('source', 'Fuente desconocida')
                    sources_by_doc[source_file].append(doc)

                for source_file, docs in sources_by_doc.items():
                    medicine_name = docs[0].metadata.get('medicine_name', 'Nombre no disponible')
                    print(f"\n📄 Documento: {medicine_name} (Archivo: {source_file})")
                    for i, doc in enumerate(docs):
                        path = doc.metadata.get('path', 'Ruta no disponible')
                        
                        # --- LÓGICA DE EXTRACCIÓN CORREGIDA ---
                        page_content = doc.page_content
                        # Usamos un separador explícito y robusto que coincide con la estructura.
                        separator = "\n---\nCONTENIDO:\n"
                        content_parts = page_content.split(separator, 1)
                        
                        # Si el separador se encontró, el texto limpio es la segunda parte.
                        # Si no, como fallback, usamos el contenido completo (nunca debería pasar).
                        extract_text = content_parts[1].strip() if len(content_parts) > 1 else page_content

                        print(f"  - Cita {i+1}:")
                        print(f"    Ruta: {path}")
                        print(f"    Extracto: \"{extract_text[:250].strip()}...\"")
            else:
                print("No se utilizaron fuentes específicas para esta respuesta.")

    except ValueError as e:
        logging.error(f"Error de configuración: {e}")
    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado: {e}", exc_info=True)

if __name__ == '__main__':
    run_chatbot()