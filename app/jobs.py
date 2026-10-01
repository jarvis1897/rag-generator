"""In-memory ingestion job registry.

Jobs live only as long as the process. That is fine for a single-instance app;
a multi-worker deployment would need a shared store (Redis, a DB table).

Progress is weighted by file size, since a file's chunk count isn't known until it
has been parsed. Within the file being processed, progress is the fraction of its
chunks embedded so far, which is where nearly all ingestion time goes.
"""

import threading
import time
import uuid

from app.schemas import FileResult, JobInfo, JobStatus

_FINISHED = {"ingested", "skipped", "failed"}
MIN_PROGRESS_FOR_ETA = 0.02  # estimates from the first few percent are noise


def file_fraction(f: FileResult) -> float:
    if f.status in _FINISHED:
        return 1.0
    if f.status == "embedding" and f.chunks:
        return f.chunks_done / f.chunks
    return 0.0


def job_progress(files: list[FileResult]) -> float:
    total = sum(max(f.size_bytes, 1) for f in files)
    return sum(max(f.size_bytes, 1) * file_fraction(f) for f in files) / total if files else 1.0


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._started: dict[str, float] = {}
        self._finished: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self, collection_id: str, files: list[tuple[str, int]]) -> str:
        """`files` is a list of (filename, size_bytes)."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobInfo(
                job_id=job_id,
                collection_id=collection_id,
                status=JobStatus.pending,
                files=[FileResult(filename=name, status="pending", size_bytes=size) for name, size in files],
            )
        return job_id

    def get(self, job_id: str) -> JobInfo | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job = job.model_copy(deep=True)
            started = self._started.get(job_id)
            finished = self._finished.get(job_id)
        job.progress = round(job_progress(job.files), 4)
        if started is not None:
            job.elapsed_seconds = round((finished or time.monotonic()) - started, 1)
            if finished is None and job.progress >= MIN_PROGRESS_FOR_ETA:
                # Linear extrapolation from elapsed time; embedding speed is roughly constant.
                job.eta_seconds = round(job.elapsed_seconds * (1 - job.progress) / job.progress, 1)
        return job

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            self._jobs[job_id].status = status
            if status == JobStatus.running:
                self._started[job_id] = time.monotonic()
            elif status in (JobStatus.done, JobStatus.failed):
                self._finished[job_id] = time.monotonic()

    def set_file_stage(self, job_id: str, index: int, status: str, chunks_done: int = 0, chunks: int = 0) -> None:
        with self._lock:
            f = self._jobs[job_id].files[index]
            f.status, f.chunks_done, f.chunks = status, chunks_done, chunks  # type: ignore[assignment]

    def set_file_result(self, job_id: str, index: int, result: FileResult) -> None:
        with self._lock:
            files = self._jobs[job_id].files
            files[index] = result.model_copy(
                update={"size_bytes": files[index].size_bytes, "chunks_done": result.chunks}
            )
