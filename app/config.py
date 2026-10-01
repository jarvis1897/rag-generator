"""Application settings, loaded from environment variables or a `.env` file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings: ingestion and query must use the same model.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, gt=0)

    # LLM
    llm_provider: Literal["anthropic"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = Field(default=4096, gt=0)
    # Server-side refusal fallback (Claude API only). Disable for providers/models that reject it.
    llm_refusal_fallback: bool = True
    # Read from env or .env. SecretStr keeps it out of reprs and logs. If unset, the
    # Anthropic SDK falls back to its own resolution (ANTHROPIC_AUTH_TOKEN, `ant auth login`).
    anthropic_api_key: SecretStr | None = None

    # Chunking (sizes are approximate tokens, see app/ingest/chunker.py)
    chunk_size: int = Field(default=250, gt=0)
    chunk_overlap: int = Field(default=40, ge=0)

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    min_relevance_score: float = Field(default=0.25, ge=0.0, le=1.0)
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    history_turns: int = Field(default=6, ge=0)

    # Storage. Empty/unset means in-memory Chroma.
    chroma_persist_dir: str | None = None

    max_upload_mb: int = Field(default=25, gt=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.chroma_persist_dir is not None and not self.chroma_persist_dir.strip():
            self.chroma_persist_dir = None
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
