"""Retrieval strategies over a single collection.

`dense`: cosine similarity search in Chroma.
`hybrid`: BM25 keyword search plus dense search, merged with reciprocal rank
fusion (RRF). BM25 helps with exact terms (product codes, names, numbers) that
small embedding models blur together.

Either way, each returned chunk's `score` is its dense cosine similarity, so the
MIN_RELEVANCE_SCORE threshold means the same thing in both modes. RRF only
decides which chunks make the top-k and in what order.
"""

import logging
import re
import threading
from dataclasses import replace
from typing import Protocol

from rank_bm25 import BM25Okapi

from app.store import DocumentStore, RetrievedChunk

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the RRF paper; dampens the weight of top ranks
CANDIDATE_MULTIPLIER = 4  # each retriever contributes top_k * this candidates to fusion


class Retriever(Protocol):
    def retrieve(self, collection_id: str, query: str, top_k: int) -> list[RetrievedChunk]: ...


class DenseRetriever:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def retrieve(self, collection_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(collection_id, query, top_k)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Merge ranked id lists. Each id scores sum(1 / (k + rank)) over the lists it appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class _Bm25Index:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.size = len(chunks)
        self.chunks = chunks
        # BM25Okapi divides by corpus size, so it can't be built over an empty collection.
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, n: int) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:n] if scores[i] > 0]


class HybridRetriever:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store
        self._indexes: dict[str, _Bm25Index] = {}
        self._lock = threading.Lock()

    def _bm25(self, collection_id: str) -> _Bm25Index:
        """BM25 index per collection, rebuilt when the chunk count changes (documents are only added)."""
        with self._lock:
            index = self._indexes.get(collection_id)
        count = self._store.count(collection_id)
        if index is None or index.size != count:
            index = _Bm25Index(self._store.all_chunks(collection_id))
            with self._lock:
                self._indexes[collection_id] = index
            logger.info("built BM25 index for %s (%d chunks)", collection_id, index.size)
        return index

    def retrieve(self, collection_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        n = top_k * CANDIDATE_MULTIPLIER
        vector = self._store.embed_query(query)
        dense = self._store.query(collection_id, query, n, vector=vector)
        keyword = self._bm25(collection_id).search(query, n)

        by_id = {c.chunk_id: c for c in keyword} | {c.chunk_id: c for c in dense}
        fused = reciprocal_rank_fusion([[c.chunk_id for c in dense], [c.chunk_id for c in keyword]])[:top_k]

        # BM25-only hits have no dense score yet; look them up so the threshold stays comparable.
        dense_ids = {c.chunk_id for c in dense}
        missing = [i for i in fused if i not in dense_ids]
        sims = self._store.similarities(collection_id, vector, missing)
        result = []
        for i in fused:
            chunk = by_id[i]
            if i in sims:
                chunk = replace(chunk, score=sims[i])
            result.append(chunk)
        return result


def create_retriever(mode: str, store: DocumentStore) -> Retriever:
    if mode == "dense":
        return DenseRetriever(store)
    if mode == "hybrid":
        return HybridRetriever(store)
    raise ValueError(f"unknown RETRIEVAL_MODE {mode!r}")
