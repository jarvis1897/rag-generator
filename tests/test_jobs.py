from app.ingest.pipeline import ingest_file
from app.jobs import JobRegistry, job_progress
from app.schemas import FileResult, JobStatus
from app.store import DocumentStore


def test_progress_weighted_by_size_and_chunks() -> None:
    files = [
        FileResult(filename="big.pdf", status="embedding", size_bytes=300, chunks=10, chunks_done=5),
        FileResult(filename="small.txt", status="ingested", size_bytes=100, chunks=2, chunks_done=2),
        FileResult(filename="next.md", status="pending", size_bytes=600),
    ]
    assert job_progress(files) == (300 * 0.5 + 100 * 1.0 + 0) / 1000


def test_registry_lifecycle_and_eta() -> None:
    jobs = JobRegistry()
    job_id = jobs.create("docset_x", [("a.txt", 100), ("b.txt", 100)])
    job = jobs.get(job_id)
    assert job.progress == 0 and job.eta_seconds is None and job.files[0].size_bytes == 100

    jobs.set_status(job_id, JobStatus.running)
    jobs.set_file_stage(job_id, 0, "embedding", chunks_done=5, chunks=10)
    job = jobs.get(job_id)
    assert job.progress == 0.25
    assert job.eta_seconds is not None and job.eta_seconds >= 0

    jobs.set_file_result(job_id, 0, FileResult(filename="a.txt", status="ingested", chunks=10))
    jobs.set_file_result(job_id, 1, FileResult(filename="b.txt", status="failed", error="bad"))
    jobs.set_status(job_id, JobStatus.failed)
    job = jobs.get(job_id)
    assert job.progress == 1.0 and job.eta_seconds is None
    assert job.files[0].chunks_done == 10 and job.files[0].size_bytes == 100


def test_ingest_reports_progress_per_batch(store: DocumentStore) -> None:
    cid = store.create_collection("x")
    text = " ".join(f"Sentence {i} about many things." for i in range(200)).encode()
    calls: list[tuple[int, int]] = []
    result = ingest_file(store, cid, "long.txt", text, chunk_size=20, chunk_overlap=2, on_progress=lambda d, t: calls.append((d, t)))
    total = result.chunks
    assert calls[0] == (0, total)  # reported as soon as chunk count is known
    assert calls[-1] == (total, total)
    assert len(calls) == 1 + -(-total // 4)  # one per embedding batch (fixture batch size 4)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)
