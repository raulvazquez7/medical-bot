import logging
from typing import List
from supabase import Client
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

class SupabaseRetriever(BaseRetriever):
    """
    A custom retriever that searches for documents in Supabase
    using the `match_documents` function.
    """
    supabase_client: Client
    embeddings_model: Embeddings
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Given a user query, converts it to an embedding and searches
        the database for the most relevant chunks.
        """
        logging.info(f"Generating embedding for the query: '{query}'")
        query_embedding = self.embeddings_model.embed_query(query)
        
        rpc_params = {
            'query_embedding': query_embedding,
            'match_count': self.top_k
        }
        
        logging.info(f"Searching for the top {self.top_k} most relevant documents in Supabase...")
        
        response = self.supabase_client.rpc('match_documents', rpc_params).execute()

        if response.data:
            logging.info(f"{len(response.data)} documents found.")
            return [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in response.data]
        
        logging.warning("No relevant documents were found.")
        return []

def get_known_medicines(client: Client) -> List[str]:
    """Consulta la BBDD para obtener la lista de medicamentos conocidos."""
    try:
        response = client.rpc('get_distinct_medicine_names', {}).execute()
        medicines = [item['medicine_name'].lower() for item in response.data]
        logging.info(f"Medicamentos cargados desde Supabase: {medicines}")
        return medicines
    except Exception as e:
        logging.error(f"No se pudo obtener la lista de medicamentos: {e}")
        return []
