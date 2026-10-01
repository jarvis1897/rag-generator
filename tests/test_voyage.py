from types import SimpleNamespace

import pytest
import voyageai.error as verr

from app import voyage
from app.config import Settings
from app.embeddings import create_embedder
from app.errors import ProviderError
from app.reranker import create_reranker
from app.store import RetrievedChunk
from app.voyage import VoyageEmbedder, VoyageReranker, token_budget_batches


class FakeVoyageClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.embed_calls: list[dict] = []
        self.rerank_calls: list[dict] = []
        self.error = error

    def embed(self, texts, model, input_type, output_dimension):
        if self.error:
            raise self.error
        self.embed_calls.append(dict(n=len(texts), model=model, input_type=input_type, dim=output_dimension))
        dim = output_dimension or 1024
        return SimpleNamespace(embeddings=[[float(len(t))] + [0.0] * (dim - 1) for t in texts], total_tokens=10)

    def rerank(self, query, documents, model, top_k):
        if self.error:
            raise self.error
        self.rerank_calls.append(dict(query=query, n=len(documents), model=model, top_k=top_k))
        # Pretend relevance = contains "gold", then shorter first.
        order = sorted(range(len(documents)), key=lambda i: ("gold" not in documents[i], len(documents[i])))
        results = [SimpleNamespace(index=i, relevance_score=1.0 - r * 0.1) for r, i in enumerate(order)]
        return SimpleNamespace(results=results[:top_k])


@pytest.fixture
def fake_client(monkeypatch) -> FakeVoyageClient:
    client = FakeVoyageClient()
    monkeypatch.setattr(voyage, "_client", lambda api_key: client)
    return client


def test_documents_and_queries_use_different_input_types(fake_client) -> None:
    emb = VoyageEmbedder("voyage-4-large", "key", dimensions=2048)
    docs = emb.embed(["a", "bb"])
    query = emb.embed_query("question?")
    assert len(docs) == 2 and len(docs[0]) == 2048 and len(query) == 2048
    assert [c["input_type"] for c in fake_client.embed_calls] == ["document", "query"]
    assert all(c["model"] == "voyage-4-large" and c["dim"] == 2048 for c in fake_client.embed_calls)


def test_model_name_includes_dimensions(fake_client) -> None:
    assert VoyageEmbedder("voyage-4-large", "key", 2048).model_name == "voyage-4-large@2048"
    assert VoyageEmbedder("voyage-4-large", "key").model_name == "voyage-4-large"


def test_requests_split_by_token_budget(fake_client) -> None:
    emb = VoyageEmbedder("voyage-4-large", "key")
    chunk = "x" * 3000  # about 1000 tokens by the conservative estimate
    vectors = emb.embed([chunk] * 250)  # about 250k tokens, over the 120k request limit
    assert len(vectors) == 250
    assert len(fake_client.embed_calls) == 3  # 108k-token budget per request
    assert sum(c["n"] for c in fake_client.embed_calls) == 250


def test_token_budget_batches_respects_text_limit() -> None:
    batches = list(token_budget_batches(["a"] * 2500, max_tokens=10**9, max_texts=1000))
    assert [len(b) for b in batches] == [1000, 1000, 500]


def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(ProviderError, match="VOYAGE_API_KEY"):
        VoyageEmbedder("voyage-4-large", None)
    with pytest.raises(ProviderError, match="VOYAGE_API_KEY"):
        VoyageReranker("rerank-2.5", "")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (verr.AuthenticationError("bad key"), "rejected the API key"),
        (verr.RateLimitError("slow down"), "rate limit"),
        (verr.ServiceUnavailableError("down"), "Voyage error"),
    ],
)
def test_sdk_errors_become_provider_errors(monkeypatch, error, message) -> None:
    monkeypatch.setattr(voyage, "_client", lambda api_key: FakeVoyageClient(error))
    with pytest.raises(ProviderError, match=message):
        VoyageEmbedder("voyage-4-large", "key").embed(["text"])
    with pytest.raises(ProviderError, match=message):
        VoyageReranker("rerank-2.5", "key").rerank("q", [RetrievedChunk("t", "f", 1, 0, "d", 0.5)], 1)


def test_reranker_maps_results_back_to_chunks(fake_client) -> None:
    chunks = [RetrievedChunk(f"filler text {i}", f"f{i}.txt", 1, 0, f"d{i}", 0.9 - i / 100) for i in range(15)]
    chunks[11] = RetrievedChunk("gold answer", "gold.txt", 7, 3, "dg", 0.5)
    ranked = VoyageReranker("rerank-2.5", "key").rerank("question", chunks, top_n=5)
    assert len(ranked) == 5
    assert ranked[0].filename == "gold.txt" and ranked[0].page == 7
    assert ranked[0].score == 0.5  # dense score kept for the threshold
    assert ranked[0].rerank_score == 1.0
    assert fake_client.rerank_calls == [dict(query="question", n=15, model="rerank-2.5", top_k=5)]


def test_factories_pick_voyage_by_name(fake_client) -> None:
    assert isinstance(create_embedder("voyage-4-large", voyage_api_key="key", dimensions=1024), VoyageEmbedder)
    assert isinstance(create_reranker("rerank-2.5", "~/.cache", "key"), VoyageReranker)
    assert create_reranker(None, "~/.cache") is None
    with pytest.raises(ValueError, match="only supported for Voyage"):
        create_embedder("BAAI/bge-m3", dimensions=2048)


def test_defaults_use_voyage() -> None:
    settings = Settings()
    assert settings.embedding_model == "voyage-4-large"
    assert settings.embedding_dimensions == 2048
    assert settings.reranker_model == "rerank-2.5"
    with pytest.raises(ValueError):
        Settings(embedding_dimensions=1500)
