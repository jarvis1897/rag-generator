"""In-memory ingestion job registry.

Jobs live only as long as the process. That is fine for a single-instance app;
a multi-worker deployment would need a shared store (Redis, a DB table).
"""

import threading
import uuid

from app.schemas import FileResult, JobInfo, JobStatus


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()

    def create(self, collection_id: str, filenames: list[str]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobInfo(
                job_id=job_id,
                collection_id=collection_id,
                status=JobStatus.pending,
                files=[FileResult(filename=f, status="pending") for f in filenames],
            )
        return job_id

    def get(self, job_id: str) -> JobInfo | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            self._jobs[job_id].status = status

    def set_file_result(self, job_id: str, index: int, result: FileResult) -> None:
        with self._lock:
            self._jobs[job_id].files[index] = result
