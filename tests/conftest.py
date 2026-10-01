import hashlib
import math
import re

import chromadb
import pytest

from app.store import DocumentStore


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. Offline and fast, good enough for tests."""

    def __init__(self, model_name: str = "fake-embedder", dim: int = 256) -> None:
        self.model_name = model_name
        self.max_input_tokens = None
        self.batch_size = 64
        self.dim = dim
        self.calls: list[int] = []  # batch sizes, to check batching
        self.query_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vectors([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return self._vectors(texts)

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chroma_client():
    client = chromadb.EphemeralClient()
    # EphemeralClient instances share state within a process; start clean.
    for col in client.list_collections():
        client.delete_collection(col if isinstance(col, str) else col.name)
    return client


@pytest.fixture
def store(chroma_client, embedder: FakeEmbedder) -> DocumentStore:
    return DocumentStore(chroma_client, embedder, embedding_batch_size=4)
