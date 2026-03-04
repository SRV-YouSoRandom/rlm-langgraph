from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen3-vl-30b-a3b-thinking"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "rlm_docs"

    # App
    app_env: str = "production"
    log_level: str = "INFO"
    max_file_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    # RLM-specific settings
    rlm_max_recursion_depth: int = 5
    rlm_repl_timeout_seconds: int = 10
    rlm_max_tokens_per_call: int = 1024
    rlm_snippet_size: int = 2000
    rlm_sandbox_mode: bool = True
    rlm_agent_max_iterations: int = 10
    rlm_fallback_to_vector: bool = True

    # Document store
    document_storage_path: str = "/app/document_storage"

    # LangGraph Checkpointer (NEW)
    checkpointer_db_path: str = "/app/checkpoints/graph.db"

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rlm-langgraph"
    enable_metrics: bool = True

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()