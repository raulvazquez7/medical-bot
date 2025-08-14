import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde .env al inicio
load_dotenv()

# --- Rutas de directorios ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MARKDOWN_PATH = os.path.join(BASE_DIR, 'data_markdown')

# --- Parámetros de Modelos ---
# Modelo multimodal para interpretar la estructura de los PDFs
PDF_PARSE_MODEL = 'gemini-2.5-flash'
# Modelo de lenguaje para el chatbot RAG
CHAT_MODEL_TO_USE = 'gemini-2.5-flash'
# Modelo de embedding. "text-embedding-3-small" es el más coste-efectivo de OpenAI.
EMBEDDINGS_MODEL = "text-embedding-3-small"

# --- Parámetros de Chunking ---
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300

# --- Credenciales (leídas desde .env) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LangSmith ---
# Para activar, establece LANGCHAIN_TRACING_V2="true" en el archivo .env
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# --- Cohere API Key ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def check_env_vars():
    """Comprueba que todas las variables de entorno necesarias para el proyecto estén cargadas."""
    
    # Lista de variables requeridas para la arquitectura actual del proyecto
    required_vars = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "COHERE_API_KEY"
    ]
    
    # Si LangSmith está habilitado, sus variables también son obligatorias
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        required_vars.extend(["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT"])

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Faltan las siguientes variables de entorno en tu archivo .env: {', '.join(missing_vars)}")
    
    print("Todas las variables de entorno necesarias están cargadas.") 