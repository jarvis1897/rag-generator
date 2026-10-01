"""Cross-encoder reranker.

A bi-encoder (the embedding model) embeds question and passage separately, which
is fast but loses detail. A cross-encoder reads the question and passage together
and scores how well the passage answers it. It is too slow to run over a whole
collection, so we retrieve a wider candidate pool cheaply and rerank only that.

The default model, ms-marco-MiniLM-L-6-v2, runs through ONNX Runtime with the
`tokenizers` library (both already Chroma dependencies), so no PyTorch is needed.
`reranker_model` in app/config.py takes any Hugging Face repo with `onnx/model.onnx` and `tokenizer.json`.
"""

import logging
import time
from dataclasses import replace
from typing import Protocol

from app.store import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512  # question + passage tokens; the model's position limit


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


class OnnxCrossEncoder:
    def __init__(self, repo_id: str, cache_dir: str, batch_size: int = 16) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from app.hf import download_onnx_model

        start = time.perf_counter()
        path = download_onnx_model(repo_id, cache_dir, ["onnx/model.onnx", "tokenizer.json"])
        model_path, tokenizer_path = str(path / "onnx" / "model.onnx"), str(path / "tokenizer.json")
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # Truncate the passage, never the question.
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LEN, strategy="only_second")
        self._tokenizer.enable_padding()
        self._batch_size = batch_size
        self.model_name = repo_id
        logger.info("loaded reranker %s in %.2fs", repo_id, time.perf_counter() - start)

    def score(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        scores: list[float] = []
        for i in range(0, len(passages), self._batch_size):
            batch = self._tokenizer.encode_batch([(query, p) for p in passages[i : i + self._batch_size]])
            feeds = {
                "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in batch], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            [logits] = self._session.run(None, feeds)
            scores.extend(float(x) for x in logits.reshape(-1))
        return scores

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        start = time.perf_counter()
        scores = self.score(query, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        logger.info("reranked %d candidates in %.2fs", len(chunks), time.perf_counter() - start)
        return [replace(c, rerank_score=s) for c, s in ranked[:top_n]]


def create_reranker(model: str | None, cache_dir: str) -> Reranker | None:
    return OnnxCrossEncoder(model, cache_dir) if model else None
