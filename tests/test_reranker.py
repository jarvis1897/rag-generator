from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.ingest.pipeline import ingest_file
from app.rag import RagPipeline
from app.retrieval import DenseRetriever
from app.store import DocumentStore, RetrievedChunk
from tests.test_rag import FakeLLM, settings


class FakeReranker:
    """Ranks chunks containing the word 'gold' first, and records what it was given."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        self.seen.append(len(chunks))
        scored = [replace(c, rerank_score=float("gold" in c.text)) for c in chunks]
        return sorted(scored, key=lambda c: c.rerank_score, reverse=True)[:top_n]


@pytest.fixture
def cid(store: DocumentStore) -> str:
    cid = store.create_collection("x")
    for i in range(20):
        word = "gold" if i == 17 else "filler"
        text = f"Valve maintenance note {i}. The valve schedule mentions {word} item {i}.".encode()
        ingest_file(store, cid, f"note{i:02}.txt", text, chunk_size=100, chunk_overlap=10)
    return cid


class StubRetriever:
    """Returns n candidates in descending dense score, with the 'gold' chunk at rank 12."""

    def __init__(self) -> None:
        self.requested: list[int] = []

    def retrieve(self, collection_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        self.requested.append(top_k)
        return [
            RetrievedChunk(f"{'gold' if i == 12 else 'filler'} {i}", f"note{i:02}.txt", 1, 0, f"d{i}", 0.9 - i * 0.01)
            for i in range(1, top_k + 1)
        ]


def test_reranker_sees_candidates_and_llm_gets_top_k() -> None:
    retriever, reranker, llm = StubRetriever(), FakeReranker(), FakeLLM()
    rag = RagPipeline(retriever, llm, settings(top_k=5, retrieval_candidates=15), reranker)
    resp = rag.answer("docset_x", "valve maintenance schedule", [])
    assert retriever.requested == [15]
    assert reranker.seen == [15]
    assert len(resp.sources) == 5
    assert resp.sources[0].filename == "note12.txt"  # rank 12 by dense, first after reranking
    assert resp.sources[0].rerank_score == 1.0
    [(_, messages)] = llm.calls
    assert messages[0]["content"].count("[note") == 5  # only the reranked top 5 reach the prompt
    assert "[1] [note12.txt p.1]" in messages[0]["content"]


def test_no_reranker_retrieves_only_top_k(store: DocumentStore, cid: str) -> None:
    resp = RagPipeline(DenseRetriever(store), FakeLLM(), settings(top_k=5)).answer(cid, "valve schedule", [])
    assert len(resp.sources) == 5
    assert all(s.rerank_score is None for s in resp.sources)


def test_below_threshold_skips_reranker_and_llm(store: DocumentStore, cid: str) -> None:
    reranker, llm = FakeReranker(), FakeLLM()
    rag = RagPipeline(DenseRetriever(store), llm, settings(min_relevance_score=0.99), reranker)
    resp = rag.answer(cid, "valve schedule", [])
    assert resp.grounded is False
    assert reranker.seen == [] and llm.calls == []


def test_candidates_must_cover_top_k() -> None:
    with pytest.raises(ValueError):
        settings(top_k=10, retrieval_candidates=5)
    settings(top_k=10, retrieval_candidates=5, reranker_model="")  # fine with reranking off


CACHE_DIR = Settings().model_cache_dir


def _cached_model() -> bool:
    return (Path(CACHE_DIR).expanduser() / "Xenova--ms-marco-MiniLM-L-6-v2" / "onnx" / "model.onnx").exists()


@pytest.mark.skipif(not _cached_model(), reason="reranker model not downloaded")
def test_onnx_cross_encoder_ranks_the_answer_first() -> None:
    from app.reranker import OnnxCrossEncoder

    def chunk(text: str) -> RetrievedChunk:
        return RetrievedChunk(text, "f.txt", 1, 0, "d", 0.5)

    chunks = [
        chunk("The Ember stove weighs 83 g and has a piezo igniter."),
        chunk("The Ember stove boils 1 litre of water in 3 minutes 20 seconds at sea level."),
        chunk("Return shipping costs $8.50 for non-members."),
    ]
    ranked = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2", CACHE_DIR).rerank(
        "How fast does the Ember stove boil water?", chunks, top_n=2
    )
    assert "boils" in ranked[0].text
    assert len(ranked) == 2 and ranked[0].rerank_score > ranked[1].rerank_score
