import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, build_state
from app.rag import NOT_FOUND_MESSAGE
from tests.test_rag import FakeLLM, settings

CATS = b"Cats are small domesticated felines. Cats purr when they are content and sleep a lot."
ROCKETS = b"Rockets burn liquid oxygen and kerosene as propellant to reach orbit."


@pytest.fixture
def client(store):
    app.state.rag_state = build_state(settings(max_upload_mb=1), store=store, llm=FakeLLM(), reranker=None)
    with TestClient(app) as c:
        yield c
    del app.state.rag_state


def create(client: TestClient, name: str) -> str:
    resp = client.post("/collections", json={"name": name})
    assert resp.status_code == 201
    return resp.json()["collection_id"]


def upload(client: TestClient, cid: str, *files: tuple[str, bytes]):
    return client.post(
        f"/collections/{cid}/documents",
        files=[("files", (name, data, "application/octet-stream")) for name, data in files],
    )


def test_full_flow(client: TestClient) -> None:
    cid = create(client, "Pets")
    resp = upload(client, cid, ("cats.txt", CATS))
    assert resp.status_code == 202
    job = client.get(f"/jobs/{resp.json()['job_id']}").json()
    assert job["status"] == "done"
    assert job["files"][0]["status"] == "ingested"

    [info] = client.get("/collections").json()
    assert info["name"] == "Pets" and info["document_count"] == 1

    answer = client.post(f"/collections/{cid}/query", json={"question": "Why do cats purr?"}).json()
    assert answer["grounded"] is True
    assert answer["sources"][0]["filename"] == "cats.txt"


def test_unsupported_type_is_400(client: TestClient) -> None:
    cid = create(client, "x")
    resp = upload(client, cid, ("cats.txt", CATS), ("photo.png", b"\x89PNG"))
    assert resp.status_code == 400
    assert ".png" in resp.json()["detail"]


def test_oversized_upload_is_413(client: TestClient) -> None:
    cid = create(client, "x")
    resp = upload(client, cid, ("big.txt", b"a" * (1024 * 1024 + 1)))
    assert resp.status_code == 413


def test_per_file_errors_reported(client: TestClient) -> None:
    cid = create(client, "x")
    resp = upload(client, cid, ("cats.txt", CATS), ("broken.pdf", b"not a pdf"), ("cats-again.txt", CATS))
    job = client.get(f"/jobs/{resp.json()['job_id']}").json()
    assert job["status"] == "failed"
    statuses = {f["filename"]: f["status"] for f in job["files"]}
    assert statuses == {"cats.txt": "ingested", "broken.pdf": "failed", "cats-again.txt": "skipped"}
    assert "PDF" in next(f for f in job["files"] if f["filename"] == "broken.pdf")["error"]


def test_unknown_collection_and_job_404(client: TestClient) -> None:
    assert upload(client, "docset_nope", ("cats.txt", CATS)).status_code == 404
    assert client.post("/collections/docset_nope/query", json={"question": "hi"}).status_code == 404
    assert client.get("/jobs/nope").status_code == 404


def test_queries_are_isolated_per_collection(client: TestClient) -> None:
    pets, space = create(client, "pets"), create(client, "space")
    upload(client, pets, ("cats.txt", CATS))
    upload(client, space, ("rockets.txt", ROCKETS))

    resp = client.post(f"/collections/{pets}/query", json={"question": "rocket propellant kerosene orbit"}).json()
    assert {s["filename"] for s in resp["sources"]} <= {"cats.txt"}
    assert resp["grounded"] is False
    assert resp["answer"] == NOT_FOUND_MESSAGE


def test_query_validation(client: TestClient) -> None:
    cid = create(client, "x")
    assert client.post(f"/collections/{cid}/query", json={"question": ""}).status_code == 422


def test_stream_endpoint(client: TestClient) -> None:
    cid = create(client, "x")
    upload(client, cid, ("cats.txt", CATS))
    with client.stream("POST", f"/collections/{cid}/query/stream", json={"question": "Why do cats purr?"}) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done" and events[-1]["grounded"] is True
