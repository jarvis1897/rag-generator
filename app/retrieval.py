"""Retrieval strategies over a single collection."""

from typing import Protocol

from app.store import DocumentStore, RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, collection_id: str, query: str, top_k: int) -> list[RetrievedChunk]: ...


class DenseRetriever:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def retrieve(self, collection_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(collection_id, query, top_k)


def create_retriever(mode: str, store: DocumentStore) -> Retriever:
    if mode == "dense":
        return DenseRetriever(store)
    raise ValueError(f"unknown RETRIEVAL_MODE {mode!r}")
