"""Thin embedding client wrapper so the model can be swapped by config.

- "all-MiniLM-L6-v2" uses Chroma's bundled ONNX build (no PyTorch needed).
- Any other name is loaded with sentence-transformers, which must be installed.
"""

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OnnxMiniLMEmbedder:
    model_name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                f"EMBEDDING_MODEL={model_name!r} needs `pip install sentence-transformers`"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def create_embedder(model_name: str) -> Embedder:
    if model_name == OnnxMiniLMEmbedder.model_name:
        return OnnxMiniLMEmbedder()
    return SentenceTransformerEmbedder(model_name)


def embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed `texts` in batches of `batch_size` (never one request per chunk)."""
    start = time.perf_counter()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed(texts[i : i + batch_size]))
    logger.info(
        "embedded %d texts in %d batch(es) in %.2fs",
        len(texts),
        -(-len(texts) // batch_size) if texts else 0,
        time.perf_counter() - start,
    )
    return vectors
