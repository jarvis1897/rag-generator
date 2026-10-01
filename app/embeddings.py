"""Thin embedding client wrapper so the model can be swapped by config.

- "voyage-*" (default "voyage-4-large"): Voyage AI's hosted API, see app/voyage.py.
- "BAAI/bge-m3": local ONNX export run with ONNX Runtime (no PyTorch needed).
  8192-token window, 1024-dim vectors. About 2.3 GB, downloaded on first use.
  Slow on CPU (about 1.5 s per 512-token chunk).
- "all-MiniLM-L6-v2": Chroma's bundled ONNX build. 256-token window, 384 dims.
- Any other name is loaded with sentence-transformers, which must be installed.
"""

import logging
import time
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str  # recorded on each collection; queries must match it
    max_input_tokens: int | None  # text beyond this is silently truncated by the model
    batch_size: int  # chunks per embed() call during ingestion

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (chunks)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Some models embed queries differently from documents."""
        ...


class _SymmetricEmbedder:
    """For models that embed queries and documents the same way."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OnnxMiniLMEmbedder(_SymmetricEmbedder):
    model_name = "all-MiniLM-L6-v2"
    max_input_tokens = 256
    batch_size = 64

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


class OnnxSentenceEmbedder(_SymmetricEmbedder):
    """Runs a sentence-transformers ONNX export that outputs a pooled `sentence_embedding`."""

    batch_size = 8  # large model, long chunks: small batches keep CPU memory in check

    def __init__(self, repo_id: str, max_input_tokens: int, cache_dir: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from app.hf import download_onnx_model

        path = download_onnx_model(repo_id, cache_dir, ["onnx/*"]) / "onnx"
        start = time.perf_counter()
        self._session = ort.InferenceSession(str(path / "model.onnx"), providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_input_tokens)
        self._tokenizer.enable_padding()
        self.model_name = repo_id
        self.max_input_tokens = max_input_tokens
        logger.info("loaded embedding model %s in %.1fs", repo_id, time.perf_counter() - start)

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        if not texts:
            return []
        encoded = self._tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
        (vectors,) = self._session.run(["sentence_embedding"], feeds)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).tolist()


# ONNX sentence-transformers exports we know the input window of.
ONNX_MODELS: dict[str, int] = {
    "BAAI/bge-m3": 8192,
}


class SentenceTransformerEmbedder(_SymmetricEmbedder):
    batch_size = 32

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                f"embedding_model={model_name!r} needs `pip install sentence-transformers`"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.max_input_tokens = self._model.max_seq_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def create_embedder(
    model_name: str,
    cache_dir: str = "~/.cache/rag-generator/models",
    voyage_api_key: str | None = None,
    dimensions: int | None = None,
) -> Embedder:
    if model_name.startswith("voyage-"):
        from app.voyage import VoyageEmbedder

        return VoyageEmbedder(model_name, voyage_api_key, dimensions)
    if dimensions:
        raise ValueError(f"embedding_dimensions is only supported for Voyage models, not {model_name!r}")
    if model_name == OnnxMiniLMEmbedder.model_name:
        return OnnxMiniLMEmbedder()
    if model_name in ONNX_MODELS:
        return OnnxSentenceEmbedder(model_name, ONNX_MODELS[model_name], cache_dir)
    return SentenceTransformerEmbedder(model_name)


ProgressCallback = Callable[[int, int], None]  # (done, total)


def embed_in_batches(
    embedder: Embedder, texts: list[str], batch_size: int, on_progress: ProgressCallback | None = None
) -> list[list[float]]:
    """Embed `texts` in batches of `batch_size` (never one request per chunk)."""
    start = time.perf_counter()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed(texts[i : i + batch_size]))
        if on_progress:
            on_progress(len(vectors), len(texts))
    logger.info(
        "embedded %d texts in %d batch(es) in %.2fs",
        len(texts),
        -(-len(texts) // batch_size) if texts else 0,
        time.perf_counter() - start,
    )
    return vectors
