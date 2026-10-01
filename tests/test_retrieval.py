from app.ingest.pipeline import ingest_file
from app.retrieval import DenseRetriever, HybridRetriever, reciprocal_rank_fusion, tokenize
from app.store import DocumentStore

DOCS = {
    "a.txt": b"The XR-7741 valve is rated for 300 bar.",
    "b.txt": b"Valves and pumps need regular maintenance every spring.",
    "c.txt": b"Cats purr when they are content.",
}


def build(store: DocumentStore) -> str:
    cid = store.create_collection("x")
    for name, data in DOCS.items():
        ingest_file(store, cid, name, data, chunk_size=100, chunk_overlap=10)
    return cid


def test_rrf_prefers_items_ranked_well_in_both() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    assert fused[0] == "b"  # rank 2 + rank 1 beats rank 1 + rank 3
    assert set(fused) == {"a", "b", "c", "d"}


def test_tokenize_keeps_codes() -> None:
    assert tokenize("Model XR-7741, v2.5!") == ["model", "xr-7741", "v2.5"]


def test_hybrid_returns_dense_scores(store: DocumentStore) -> None:
    cid = build(store)
    hybrid = HybridRetriever(store).retrieve(cid, "XR-7741 rating", top_k=3)
    assert hybrid[0].filename == "a.txt"
    dense = {c.chunk_id: c.score for c in DenseRetriever(store).retrieve(cid, "XR-7741 rating", top_k=10)}
    for c in hybrid:
        assert abs(c.score - dense[c.chunk_id]) < 1e-6  # threshold semantics match dense mode


def test_hybrid_is_scoped_and_index_refreshes(store: DocumentStore) -> None:
    cid = build(store)
    other = store.create_collection("other")
    ingest_file(store, other, "d.txt", b"Rockets burn kerosene.", chunk_size=100, chunk_overlap=10)
    retriever = HybridRetriever(store)
    assert {c.filename for c in retriever.retrieve(cid, "kerosene rockets", top_k=5)} <= set(DOCS)

    ingest_file(store, cid, "e.txt", b"Kerosene storage rules for the depot.", chunk_size=100, chunk_overlap=10)
    assert "e.txt" in {c.filename for c in retriever.retrieve(cid, "kerosene", top_k=5)}


def test_hybrid_empty_collection(store: DocumentStore) -> None:
    cid = store.create_collection("empty")
    assert HybridRetriever(store).retrieve(cid, "anything", top_k=3) == []
