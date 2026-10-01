"""Application configuration.

Two kinds of configuration, each with exactly one home:

- `Settings`: every tunable value lives here, in this file, as a default. Edit this
  file to change them. They are NOT read from environment variables or `.env`, so a
  value can never be defined in two places that disagree.
- `Secrets`: API keys only, read from environment variables or `.env`. They never
  appear in source code or logs.

If a setting name shows up in the environment or `.env`, it is ignored and a warning
is logged at startup.
"""

import logging
import os
from functools import lru_cache
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ENV_FILE = ".env"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Embeddings ---
    # Ingestion and query must use the same model; collections record which one built them.
    # bge-m3: 8192-token input window, 1024-dim vectors, strong on long formal text (legal,
    # contracts, policies). Runs locally via ONNX. "all-MiniLM-L6-v2" is a small, fast
    # alternative (256-token window, 384 dims). Changing the model requires re-ingesting.
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = Field(default=8, gt=0)  # long chunks: keep batches small for CPU memory

    # --- LLM ---
    llm_provider: Literal["anthropic"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = Field(default=4096, gt=0)
    # Server-side refusal fallback (Claude API only). Disable for providers/models that reject it.
    llm_refusal_fallback: bool = True

    # --- Chunking (approximate tokens, see app/ingest/chunker.py) ---
    # Legal text needs whole clauses with their qualifiers in one chunk. 512 fits easily in
    # bge-m3's window and roughly matches the reranker's 512-token limit, so the reranker
    # sees (nearly) the whole chunk.
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)

    # --- Retrieval ---
    # top_k chunks go to the LLM. With a reranker, retrieval_candidates chunks are retrieved
    # first and the reranker picks the best top_k of them.
    top_k: int = Field(default=5, gt=0)
    retrieval_candidates: int = Field(default=15, gt=0)
    # Hugging Face repo with an ONNX cross-encoder. None disables reranking.
    reranker_model: str | None = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Cosine similarity gate, calibrated per embedding model (see README). Re-check it
    # whenever embedding_model changes.
    min_relevance_score: float = Field(default=0.42, ge=0.0, le=1.0)
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    history_turns: int = Field(default=6, ge=0)

    # --- Storage and uploads ---
    # Relative paths resolve from the working directory. None means in-memory Chroma
    # (everything is lost on restart).
    chroma_persist_dir: str | None = "./chroma_data"
    max_upload_mb: int = Field(default=25, gt=0)
    # Downloaded embedding and reranker models (about 2.4 GB with the defaults).
    model_cache_dir: str = "~/.cache/rag-generator/models"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.reranker_model and self.retrieval_candidates < self.top_k:
            raise ValueError("retrieval_candidates must be at least top_k")
        return self


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # SecretStr keeps it out of reprs and logs. If unset, the Anthropic SDK falls back to
    # its own resolution (ANTHROPIC_AUTH_TOKEN, `ant auth login`).
    anthropic_api_key: SecretStr | None = None


def _warn_on_ignored_overrides() -> None:
    """Settings live only in this file; flag values someone tried to set elsewhere."""
    names = {name.upper() for name in Settings.model_fields}
    file_values = dotenv_values(ENV_FILE) if os.path.exists(ENV_FILE) else {}
    for source, keys in (("environment", os.environ.keys()), (ENV_FILE, file_values.keys())):
        ignored = sorted(names & {k.upper() for k in keys})
        if ignored:
            logger.warning(
                "ignoring %s set in %s: settings are configured only in app/config.py",
                ", ".join(ignored),
                source,
            )


@lru_cache
def get_settings() -> Settings:
    _warn_on_ignored_overrides()
    return Settings()


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
