import pytest

from app.ingest.pipeline import ingest_file
from app.store import CollectionNotFound, DocumentStore, EmbeddingModelMismatch
from tests.conftest import FakeEmbedder

CATS = b"Cats are small domesticated felines. Cats purr when they are content."
ROCKETS = b"Rockets use liquid oxygen and kerosene as propellant to reach orbit."


def ingest(store: DocumentStore, cid: str, name: str, data: bytes, size: int = 50):
    return ingest_file(store, cid, name, data, chunk_size=size, chunk_overlap=5)


def test_create_and_list(store: DocumentStore) -> None:
    cid = store.create_collection("Pets")
    ingest(store, cid, "cats.txt", CATS)
    [info] = store.list_collections()
    assert info["collection_id"] == cid
    assert info["name"] == "Pets"
    assert info["document_count"] == 1
    assert info["chunk_count"] >= 1
    assert info["embedding_model"] == "fake-embedder"


def test_metadata_is_stored(store: DocumentStore) -> None:
    cid = store.create_collection("x")
    ingest(store, cid, "cats.txt", CATS)
    [hit] = store.query(cid, "cats purr", top_k=1)
    assert hit.filename == "cats.txt"
    assert hit.page == 1
    assert hit.chunk_index == 0
    assert len(hit.doc_id) == 16
    assert 0 < hit.score <= 1.0


def test_duplicate_file_is_skipped(store: DocumentStore) -> None:
    cid = store.create_collection("x")
    assert ingest(store, cid, "cats.txt", CATS).status == "ingested"
    # Same bytes under a different name is still a duplicate.
    result = ingest(store, cid, "copy-of-cats.txt", CATS)
    assert result.status == "skipped"
    assert store.list_collections()[0]["document_count"] == 1


def test_same_file_allowed_in_other_collection(store: DocumentStore) -> None:
    a, b = store.create_collection("a"), store.create_collection("b")
    assert ingest(store, a, "cats.txt", CATS).status == "ingested"
    assert ingest(store, b, "cats.txt", CATS).status == "ingested"


def test_collection_isolation(store: DocumentStore) -> None:
    pets, space = store.create_collection("pets"), store.create_collection("space")
    ingest(store, pets, "cats.txt", CATS)
    ingest(store, space, "rockets.txt", ROCKETS)

    pet_hits = store.query(pets, "rocket propellant orbit", top_k=10)
    space_hits = store.query(space, "cats purr", top_k=10)
    assert {h.filename for h in pet_hits} == {"cats.txt"}
    assert {h.filename for h in space_hits} == {"rockets.txt"}


def test_embedding_batches(store: DocumentStore, embedder: FakeEmbedder) -> None:
    cid = store.create_collection("x")
    text = " ".join(f"Sentence {i} about many things." for i in range(200)).encode()
    result = ingest(store, cid, "long.txt", text, size=20)
    assert result.chunks > 4
    assert max(embedder.calls) <= 4  # batch size from the fixture
    assert len(embedder.calls) == -(-result.chunks // 4)


def test_embedding_model_mismatch_rejected(chroma_client, store: DocumentStore) -> None:
    cid = store.create_collection("x")
    ingest(store, cid, "cats.txt", CATS)
    other = DocumentStore(chroma_client, FakeEmbedder(model_name="different-model"))
    with pytest.raises(EmbeddingModelMismatch):
        other.query(cid, "cats", top_k=3)
    with pytest.raises(EmbeddingModelMismatch):
        ingest(other, cid, "rockets.txt", ROCKETS)


def test_unknown_collection(store: DocumentStore) -> None:
    with pytest.raises(CollectionNotFound):
        store.query("docset_doesnotexist", "x", top_k=1)
    assert not store.exists("not_ours")


def test_empty_collection_query(store: DocumentStore) -> None:
    cid = store.create_collection("empty")
    assert store.query(cid, "anything", top_k=5) == []
