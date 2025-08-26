import os
from dotenv import load_dotenv

# Load environment variables from .env at the start
load_dotenv()

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MARKDOWN_PATH = os.path.join(BASE_DIR, 'data_markdown')

# --- Model Parameters ---
# Multimodal model to interpret the structure of the PDFs
PDF_PARSE_MODEL = 'gemini-1.5-flash'

# [MODIFICADO] Modelo principal para el agente ReAct (razonamiento complejo)
# Se recomienda un modelo potente como 'gpt-4o' o 'gemini-1.5-pro'
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")

# [NUEVO] Modelo rápido y económico para tareas de clasificación y resumen
# Se recomienda un modelo rápido como 'gemini-1.5-flash'
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-2.5-flash")

# Embeddings provider ("openai" or "google")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "google")
# Embedding model. "text-embedding-3-small" for OpenAI or "models/gemini-embedding-001" for Google.
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "models/gemini-embedding-001")

# --- Chunking Parameters ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Evaluation Parameters ---
# Enable or disable the re-ranking step during evaluation
EVAL_USE_RERANKER = os.getenv("EVAL_USE_RERANKER", "True").lower() in ('true', '1', 't')
# Number of documents to retrieve initially before re-ranking
EVAL_INITIAL_K = int(os.getenv("EVAL_INITIAL_K", "20"))
# Final number of documents to consider after re-ranking
EVAL_FINAL_K = int(os.getenv("EVAL_FINAL_K", "5"))

# --- Credentials (read from .env) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Database Connection for Checkpoints ---
POSTGRES_CONN_STR = os.getenv("POSTGRES_CONN_STR")

# --- LangSmith ---
# To enable, set LANGCHAIN_TRACING_V2="true" in the .env file
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# --- Cohere API Key ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def check_env_vars():
    """Checks that all necessary environment variables for the project are loaded."""
    
    # List of required variables for the current project architecture
    required_vars = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "COHERE_API_KEY",
        "POSTGRES_CONN_STR"
    ]
    
    # If LangSmith is enabled, its variables are also mandatory
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        required_vars.extend(["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT"])

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"The following environment variables are missing from your .env file: {', '.join(missing_vars)}")
    
    print("All necessary environment variables are loaded.") 