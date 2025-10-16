import os
from pydantic_settings import BaseSettings
from pydantic import Field



class Settings(BaseSettings):
    """
    Application settings with validation.
    Uses Pydantic Settings for automatic env var loading and type validation.
    """

    # --- Directory Paths ---
    base_dir: str = Field(
        default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        description="Base directory of the project",
    )

    @property
    def data_path(self) -> str:
        """Path to data directory."""
        return os.path.join(self.base_dir, "data")

    @property
    def markdown_path(self) -> str:
        """Path to markdown data directory."""
        return os.path.join(self.base_dir, "data_markdown")

    # --- Model Configuration ---
    pdf_parse_model: str = Field(
        default="gemini-2.5-flash",
        description="Multimodal model for PDF parsing",
    )

    agent_model: str = Field(
        default="gpt-4o",
        description="Main agent LLM (ReAct reasoning)",
    )

    router_model: str = Field(
        default="gemini-2.5-flash",
        description="Fast LLM for classification and summarization",
    )

    embeddings_provider: str = Field(
        default="google",
        description="Embeddings provider (openai or google)",
    )

    embeddings_model: str = Field(
        default="models/gemini-embedding-001",
        description="Embeddings model name",
    )

    # --- LLM Service Configuration ---
    llm_timeout: int = Field(
        default=30,
        description="Timeout in seconds for LLM calls",
        ge=5,
        le=120,
    )
    llm_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed LLM calls",
        ge=1,
        le=5,
    )
    llm_rate_limit: int = Field(
        default=3,
        description="Maximum concurrent LLM requests (rate limiting)",
        ge=1,
        le=10,
    )
    embeddings_cache_size: int = Field(
        default=100,
        description="LRU cache size for embeddings queries",
        ge=0,
        le=1000,
    )

    # --- Agent Configuration ---
    max_react_iterations: int = Field(
        default=10,
        description="Maximum ReAct agent iterations to prevent infinite loops",
        ge=1,
        le=20,
    )

    # --- Chunking Parameters ---
    chunk_size: int = Field(default=800, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")

    # --- Evaluation Parameters ---
    eval_use_reranker: bool = Field(
        default=True, description="Enable re-ranking in evaluation"
    )
    eval_initial_k: int = Field(
        default=20, description="Initial documents to retrieve"
    )
    eval_final_k: int = Field(default=5, description="Final documents after re-ranking")

    # --- API Keys (Required) ---
    google_api_key: str = Field(..., description="Google API key for Gemini")
    openai_api_key: str = Field(..., description="OpenAI API key")
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(..., description="Supabase service key")
    cohere_api_key: str = Field(..., description="Cohere API key for re-ranking")
    postgres_conn_str: str = Field(
        ..., description="PostgreSQL connection string for checkpoints"
    )

    # --- LangSmith (Optional) ---
    langchain_tracing_v2: str = Field(
        default="false", description="Enable LangSmith tracing"
    )
    langchain_api_key: str | None = Field(
        default=None, description="LangSmith API key"
    )
    langchain_project: str | None = Field(
        default=None, description="LangSmith project name"
    )

    class Config:
        """Pydantic config."""

        env_file = os.getenv("DOTENV_PATH", ".env")
        case_sensitive = False
        extra = "ignore"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get or create settings instance.
    Singleton pattern for consistent configuration.

    Returns:
        Settings instance

    Raises:
        ValidationError: If required env vars are missing or invalid
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Expose settings as module-level variables for backward compatibility
# Only expose non-sensitive configuration values
settings = get_settings()

# Directory paths
BASE_DIR = settings.base_dir
DATA_PATH = settings.data_path
MARKDOWN_PATH = settings.markdown_path

# Model parameters
PDF_PARSE_MODEL = settings.pdf_parse_model
AGENT_MODEL = settings.agent_model
ROUTER_MODEL = settings.router_model
EMBEDDINGS_PROVIDER = settings.embeddings_provider
EMBEDDINGS_MODEL = settings.embeddings_model

# LLM Service parameters
LLM_TIMEOUT = settings.llm_timeout
LLM_MAX_RETRIES = settings.llm_max_retries
LLM_RATE_LIMIT = settings.llm_rate_limit
EMBEDDINGS_CACHE_SIZE = settings.embeddings_cache_size
MAX_REACT_ITERATIONS = settings.max_react_iterations

# Chunking parameters
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

# Evaluation parameters
EVAL_USE_RERANKER = settings.eval_use_reranker
EVAL_INITIAL_K = settings.eval_initial_k
EVAL_FINAL_K = settings.eval_final_k

# NOTE: API keys and credentials are NOT exposed globally for security.
# Access them via get_settings() when needed:
#   from src.config import get_settings
#   settings = get_settings()
#   api_key = settings.openai_api_key


def check_env_vars():
    """
    Validates that all necessary environment variables are loaded.
    Now handled automatically by Pydantic Settings.

    Raises:
        ValidationError: If required variables are missing
    """
    try:
        get_settings()
        print("✅ All necessary environment variables are loaded and validated.")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise 