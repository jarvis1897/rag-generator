"""Chroma wrapper: one collection per document set.

Embeddings are computed by our own `Embedder` and passed in explicitly, so the
collection never embeds anything itself. The embedding model name is stored in
the collection metadata and every write and query checks it.
"""

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.embeddings import Embedder, ProgressCallback, embed_in_batches
from app.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "docset_"


class CollectionNotFound(KeyError):
    pass


class EmbeddingModelMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    page: int
    chunk_index: int
    doc_id: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant
    rerank_score: float | None = None  # cross-encoder logit, set only when reranking is on

    @property
    def chunk_id(self) -> str:
        return chunk_id(self.doc_id, self.chunk_index)


def chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index}"


def create_client(persist_dir: str | None) -> ClientAPI:
    if persist_dir:
        logger.info("using persistent Chroma at %s", persist_dir)
        return chromadb.PersistentClient(path=persist_dir)
    logger.info("using in-memory Chroma (data is lost on restart)")
    return chromadb.EphemeralClient()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DocumentStore:
    def __init__(self, client: ClientAPI, embedder: Embedder, embedding_batch_size: int | None = None) -> None:
        self._client = client
        self._embedder = embedder
        self._batch_size = embedding_batch_size or embedder.batch_size

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # --- collections ---

    def create_collection(self, name: str) -> str:
        collection_id = COLLECTION_PREFIX + uuid.uuid4().hex[:16]
        self._client.create_collection(
            name=collection_id,
            metadata={
                "hnsw:space": "cosine",
                "display_name": name,
                "embedding_model": self._embedder.model_name,
            },
        )
        return collection_id

    def list_collections(self) -> list[dict[str, Any]]:
        result = []
        for col in self._client.list_collections():
            # Older Chroma versions return names, newer ones return Collection objects.
            col = col if isinstance(col, Collection) else self._client.get_collection(col)
            if not col.name.startswith(COLLECTION_PREFIX):
                continue
            meta = col.metadata or {}
            result.append(
                {
                    "collection_id": col.name,
                    "name": meta.get("display_name", col.name),
                    "embedding_model": meta.get("embedding_model", "unknown"),
                    "document_count": len(self._doc_ids(col)),
                    "chunk_count": col.count(),
                }
            )
        return result

    def _get(self, collection_id: str) -> Collection:
        if not collection_id.startswith(COLLECTION_PREFIX):
            raise CollectionNotFound(collection_id)
        try:
            return self._client.get_collection(collection_id)
        except Exception as exc:  # Chroma raises different types across versions
            raise CollectionNotFound(collection_id) from exc

    def _checked(self, collection_id: str) -> Collection:
        col = self._get(collection_id)
        stored = (col.metadata or {}).get("embedding_model")
        if stored != self._embedder.model_name:
            raise EmbeddingModelMismatch(
                f"collection was built with embedding model {stored!r} but "
                f"embedding_model in app/config.py is {self._embedder.model_name!r}; re-ingest into a new collection"
            )
        return col

    def exists(self, collection_id: str) -> bool:
        try:
            self._get(collection_id)
        except CollectionNotFound:
            return False
        return True

    @staticmethod
    def _doc_ids(col: Collection) -> set[str]:
        metadatas = col.get(include=["metadatas"])["metadatas"] or []
        return {str(m["doc_id"]) for m in metadatas if m and "doc_id" in m}

    # --- documents ---

    def has_document(self, collection_id: str, content_hash: str) -> bool:
        col = self._get(collection_id)
        found = col.get(where={"content_hash": content_hash}, limit=1, include=[])
        return bool(found["ids"])

    def add_chunks(
        self,
        collection_id: str,
        filename: str,
        content_hash: str,
        chunks: list[Chunk],
        on_progress: ProgressCallback | None = None,
    ) -> int:
        """Embed and store chunks for one document. Returns the number of chunks added."""
        if not chunks:
            return 0
        col = self._checked(collection_id)
        doc_id = content_hash[:16]
        vectors = embed_in_batches(self._embedder, [c.text for c in chunks], self._batch_size, on_progress)
        col.add(
            ids=[chunk_id(doc_id, c.chunk_index) for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            # Chroma metadata values must be str/int/float/bool: no None, no lists.
            metadatas=[
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": int(c.page),
                    "chunk_index": int(c.chunk_index),
                    "content_hash": content_hash,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def count(self, collection_id: str) -> int:
        return self._get(collection_id).count()

    def all_chunks(self, collection_id: str) -> list[RetrievedChunk]:
        """Every chunk in a collection (score 0). Used to build the BM25 index."""
        col = self._checked(collection_id)
        got = col.get(include=["documents", "metadatas"])
        return [
            _to_chunk(doc, meta, 0.0)
            for doc, meta in zip(got["documents"] or [], got["metadatas"] or [])
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def query(
        self, collection_id: str, text: str, top_k: int, vector: list[float] | None = None
    ) -> list[RetrievedChunk]:
        """Dense search within exactly one collection."""
        col = self._checked(collection_id)
        if col.count() == 0:
            return []
        vector = vector if vector is not None else self.embed_query(text)
        res = col.query(
            query_embeddings=[vector],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            _to_chunk(doc, meta, 1.0 - float(dist))  # cosine distance -> similarity
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def similarities(self, collection_id: str, vector: list[float], ids: list[str]) -> dict[str, float]:
        """Cosine similarity between `vector` and the stored embeddings of `ids`."""
        if not ids:
            return {}
        got = self._checked(collection_id).get(ids=ids, include=["embeddings"])
        return {i: _cosine(vector, list(e)) for i, e in zip(got["ids"], got["embeddings"])}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _to_chunk(doc: str, meta: Any, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=doc,
        filename=str(meta["filename"]),
        page=int(meta["page"]),
        chunk_index=int(meta["chunk_index"]),
        doc_id=str(meta["doc_id"]),
        score=score,
    )
