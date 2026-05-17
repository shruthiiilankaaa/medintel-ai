"""
app/core/config.py
Centralised settings loaded from .env
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"

    # Vector DB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "medintel_docs"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_docs: int = 4
    confidence_threshold: float = 0.35

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Frontend
    streamlit_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
