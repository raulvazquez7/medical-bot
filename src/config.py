import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# --- Rutas del Proyecto ---
# Obtenemos la ruta raíz del proyecto para construir rutas absolutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
MARKDOWN_PATH = os.path.join(PROJECT_ROOT, 'data_markdown')

# --- Claves de API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- Modelos de IA ---
# Modelo para parsear los PDFs. 'gemini-1.5-flash-latest' es rápido y coste-efectivo.
PDF_PARSE_MODEL = 'gemini-2.5-flash'
# Modelo de embeddings. 'text-embedding-3-small' es el estándar de OpenAI.
EMBEDDINGS_MODEL = "text-embedding-3-small"

# --- Modelos de Chat ---
# Define los modelos que se pueden usar para el chat.
# Cambia CHAT_MODEL_TO_USE para seleccionar cuál usar.
OPENAI_CHAT_MODEL = "gpt-4.1"
GOOGLE_CHAT_MODEL = "gemini-2.5-flash"

# Modelo a utilizar para la generación de respuestas en el chat.
CHAT_MODEL_TO_USE = GOOGLE_CHAT_MODEL

# --- Parámetros de Chunking ---
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300

def check_env_vars():
    """Comprueba que todas las variables de entorno necesarias estén definidas."""
    required_vars = {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
    }
    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Faltan variables de entorno: {', '.join(missing_vars)}")
    else:
        print("Todas las variables de entorno necesarias están cargadas.") 