import logging
from typing import List
from supabase import Client
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config

class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Wrapper class around GoogleGenerativeAIEmbeddings to enforce
    a specific output dimensionality and the appropriate task_type in each call.
    """
    output_dim: int = 1536

    def embed_query(self, text: str, **kwargs) -> List[float]:
        """Overrides embed_query to add parameters optimized for search."""
        # We remove parameters to avoid conflicts if they are passed explicitly.
        kwargs.pop("output_dimensionality", None)
        kwargs.pop("task_type", None)
        return super().embed_query(
            text=text, 
            output_dimensionality=self.output_dim, 
            task_type="retrieval_query", 
            **kwargs
        )

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Overrides embed_documents to add parameters optimized for storage."""
        kwargs.pop("output_dimensionality", None)
        kwargs.pop("task_type", None)
        return super().embed_documents(
            texts=texts, 
            output_dimensionality=self.output_dim, 
            task_type="retrieval_document", 
            **kwargs
        )

def get_known_medicines(supabase_client: Client) -> List[str]:
    """Retrieves the list of unique medicine names from the database."""
    try:
        response = supabase_client.table('documents').select("metadata->>medicine_name").execute()
        if response.data:
            # We use a set to get unique names and then convert it to a sorted list
            unique_medicines = sorted(list(set(item['medicine_name'] for item in response.data)))
            logging.info(f"Known medicines found in the DB: {unique_medicines}")
            return unique_medicines
    except Exception as e:
        logging.error(f"Could not retrieve the list of medicines: {e}")
        return []

def get_embeddings_model() -> Embeddings:
    """
    Centralized factory to get the configured embeddings model.
    """
    provider = config.EMBEDDINGS_PROVIDER.lower()
    logging.info(f"Initializing embeddings model from: {provider}")
    
    if provider == "google":
        # Now we use our custom wrapper class.
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
        raise ValueError(f"Unsupported embeddings provider: {provider}")
