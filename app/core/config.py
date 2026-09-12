from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Granimals Knowledge Assistant"
    app_env: str = "development"
    log_level: str = "INFO"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_vision_model: str = "qwen2.5-vision:0.5b"

    ollama_embedding_model: str = "nomic-embed-text:latest"
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    reranker_model: str = "BAAI/bge-reranker-base"

    chunk_size: int = Field(
        default=1500,
        ge=500,
        le=5000,
    )

    chunk_overlap: int = Field(
        default=250,
        ge=0,
        le=1000,
    )

    dense_top_k: int = Field(
        default=15,
        ge=1,
        le=100,
    )

    sparse_top_k: int = Field(
        default=15,
        ge=1,
        le=100,
    )

    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
    )

    max_context_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    max_context_tokens: int = Field(
        default=5000,
        ge=500,
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    enable_vision: bool = True
    enable_ocr: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()