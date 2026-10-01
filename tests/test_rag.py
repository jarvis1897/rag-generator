from collections.abc import Iterator

import pytest

from app.config import Settings
from app.ingest.pipeline import ingest_file
from app.rag import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, RagPipeline, build_context
from app.retrieval import DenseRetriever
from app.schemas import ChatTurn
from app.store import DocumentStore, RetrievedChunk

CATS = b"Cats are small domesticated felines. Cats purr when they are content and sleep a lot."


class FakeLLM:
    """Records calls and returns canned replies."""

    def __init__(self, answer: str = "Cats purr when content [cats.txt p.1].", rewrite: str = "") -> None:
        self.answer = answer
        self.rewrite = rewrite
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def complete(self, system, messages, max_tokens, effort=None) -> str:
        return "".join(self.stream(system, messages, max_tokens, effort))

    def stream(self, system, messages, max_tokens, effort=None) -> Iterator[str]:
        self.calls.append((system, messages))
        text = self.answer if system == SYSTEM_PROMPT else self.rewrite
        yield from text.split(" ")[:1] + [" " + w for w in text.split(" ")[1:]]


def settings(**overrides) -> Settings:
    base = dict(min_relevance_score=0.3, top_k=3, chroma_persist_dir=None, _env_file=None)
    return Settings(**{**base, **overrides})


@pytest.fixture
def cid(store: DocumentStore) -> str:
    cid = store.create_collection("pets")
    ingest_file(store, cid, "cats.txt", CATS, chunk_size=100, chunk_overlap=10)
    return cid


def test_below_threshold_refuses_without_llm_call(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM()
    rag = RagPipeline(DenseRetriever(store), llm, settings(min_relevance_score=0.99))
    resp = rag.answer(cid, "What is the boiling point of mercury?", [])
    assert resp.grounded is False
    assert resp.answer == NOT_FOUND_MESSAGE
    assert llm.calls == []  # rule 1: the LLM is never called
    assert resp.sources  # rule 4: sources are still returned


def test_unrelated_question_is_below_default_threshold(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM()
    rag = RagPipeline(DenseRetriever(store), llm, settings())
    resp = rag.answer(cid, "quantum chromodynamics gluon", [])
    assert resp.grounded is False
    assert llm.calls == []


def test_empty_collection_refuses(store: DocumentStore) -> None:
    empty = store.create_collection("empty")
    llm = FakeLLM()
    resp = RagPipeline(DenseRetriever(store), llm, settings()).answer(empty, "cats?", [])
    assert resp.grounded is False and resp.sources == [] and llm.calls == []


def test_grounded_answer(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM()
    resp = RagPipeline(DenseRetriever(store), llm, settings()).answer(cid, "Why do cats purr?", [])
    assert resp.grounded is True
    assert resp.answer == "Cats purr when content [cats.txt p.1]."
    assert resp.sources[0].filename == "cats.txt" and resp.sources[0].index == 1
    [(system, messages)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert "[1] [cats.txt p.1]" in messages[0]["content"]


def test_model_refusal_marks_ungrounded(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM(answer=NOT_FOUND_MESSAGE + " The documents only cover cat behaviour.")
    resp = RagPipeline(DenseRetriever(store), llm, settings()).answer(cid, "How do cats purr?", [])
    assert resp.grounded is False
    assert resp.sources


def test_follow_up_is_rewritten_and_history_not_sent_to_generation(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM(rewrite="Why do cats purr?")
    history = [ChatTurn(role="user", content="Tell me about cats"), ChatTurn(role="assistant", content="SECRET-HISTORY")]
    resp = RagPipeline(DenseRetriever(store), llm, settings()).answer(cid, "Why do they purr?", history)
    assert resp.standalone_question == "Why do cats purr?"
    rewrite_call, gen_call = llm.calls
    assert "SECRET-HISTORY" in rewrite_call[1][0]["content"]
    assert "SECRET-HISTORY" not in gen_call[1][0]["content"]  # rule 5: answer from context only
    assert "Why do cats purr?" in gen_call[1][0]["content"]


def test_no_history_skips_rewrite(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM()
    RagPipeline(DenseRetriever(store), llm, settings()).answer(cid, "Why do cats purr?", [])
    assert len(llm.calls) == 1


def test_stream_events(store: DocumentStore, cid: str) -> None:
    rag = RagPipeline(DenseRetriever(store), FakeLLM(), settings())
    events = list(rag.answer_events(cid, "Why do cats purr?", []))
    assert events[0]["type"] == "sources"
    assert events[-1] == {"type": "done", "answer": "Cats purr when content [cats.txt p.1].", "grounded": True}
    assert "".join(e["text"] for e in events if e["type"] == "token") == events[-1]["answer"]


def test_stream_events_refusal(store: DocumentStore, cid: str) -> None:
    llm = FakeLLM()
    rag = RagPipeline(DenseRetriever(store), llm, settings(min_relevance_score=0.99))
    events = list(rag.answer_events(cid, "anything", []))
    assert events[-1]["grounded"] is False and llm.calls == []


def test_context_numbering() -> None:
    chunks = [
        RetrievedChunk("alpha", "a.pdf", 3, 0, "d1", 0.9),
        RetrievedChunk("beta", "b.md", 1, 4, "d2", 0.8),
    ]
    assert build_context(chunks) == "[1] [a.pdf p.3]\nalpha\n\n[2] [b.md p.1]\nbeta"
