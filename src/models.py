import logging
from typing import List
from supabase import Client
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config

class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Clase envoltorio (wrapper) sobre GoogleGenerativeAIEmbeddings para forzar
    una dimensionalidad de salida específica y el task_type adecuado en cada llamada.
    """
    output_dim: int = 1536

    def embed_query(self, text: str, **kwargs) -> List[float]:
        """Sobrescribe embed_query para añadir parámetros optimizados para búsquedas."""
        # Eliminamos parámetros para evitar conflictos si se pasan explícitamente.
        kwargs.pop("output_dimensionality", None)
        kwargs.pop("task_type", None)
        return super().embed_query(
            text=text, 
            output_dimensionality=self.output_dim, 
            task_type="retrieval_query", 
            **kwargs
        )

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Sobrescribe embed_documents para añadir parámetros optimizados para almacenamiento."""
        kwargs.pop("output_dimensionality", None)
        kwargs.pop("task_type", None)
        return super().embed_documents(
            texts=texts, 
            output_dimensionality=self.output_dim, 
            task_type="retrieval_document", 
            **kwargs
        )

def get_known_medicines(supabase_client: Client) -> List[str]:
    """Recupera la lista de nombres de medicamentos únicos de la base de datos."""
    try:
        response = supabase_client.table('documents').select("metadata->>medicine_name").execute()
        if response.data:
            # Usamos un set para obtener nombres únicos y luego lo convertimos a lista ordenada
            unique_medicines = sorted(list(set(item['medicine_name'] for item in response.data)))
            logging.info(f"Medicamentos conocidos encontrados en la BD: {unique_medicines}")
            return unique_medicines
    except Exception as e:
        logging.error(f"No se pudo recuperar la lista de medicamentos: {e}")
        return []

def get_embeddings_model() -> Embeddings:
    """
    Fábrica centralizada para obtener el modelo de embeddings configurado.
    """
    provider = config.EMBEDDINGS_PROVIDER.lower()
    logging.info(f"Inicializando modelo de embeddings de: {provider}")
    
    if provider == "google":
        # Ahora usamos nuestra clase envoltorio personalizada.
        return CustomGoogleGenerativeAIEmbeddings(
            google_api_key=config.GOOGLE_API_KEY, 
            model=config.EMBEDDINGS_MODEL
        )
        
    elif provider == "openai":
        return OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY, 
            model=config.EMBEDDINGS_MODEL
        )
        
    else:
        raise ValueError(f"Proveedor de embeddings no soportado: {provider}")
