"""Thin embedding client wrapper so the model can be swapped by config.

- "BAAI/bge-m3" (default): ONNX export run with ONNX Runtime (no PyTorch needed).
  8192-token window, 1024-dim vectors. About 2.3 GB, downloaded on first use.
- "all-MiniLM-L6-v2": Chroma's bundled ONNX build. 256-token window, 384 dims.
- Any other name is loaded with sentence-transformers, which must be installed.
"""

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    max_input_tokens: int | None  # text beyond this is silently truncated by the model

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OnnxMiniLMEmbedder:
    model_name = "all-MiniLM-L6-v2"
    max_input_tokens = 256

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


class OnnxSentenceEmbedder:
    """Runs a sentence-transformers ONNX export that outputs a pooled `sentence_embedding`."""

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


class SentenceTransformerEmbedder:
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


def create_embedder(model_name: str, cache_dir: str = "~/.cache/rag-generator/models") -> Embedder:
    if model_name == OnnxMiniLMEmbedder.model_name:
        return OnnxMiniLMEmbedder()
    if model_name in ONNX_MODELS:
        return OnnxSentenceEmbedder(model_name, ONNX_MODELS[model_name], cache_dir)
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
