"""FastAPI app and routes."""

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.embeddings import create_embedder
from app.ingest.parsers import UnsupportedFileType, check_supported
from app.ingest.pipeline import ingest_file
from app.jobs import JobRegistry
from app.llm import LLMClient, LLMError, create_llm
from app.rag import RagPipeline
from app.retrieval import create_retriever
from app.schemas import (
    CollectionCreate,
    CollectionCreated,
    CollectionInfo,
    FileResult,
    JobCreated,
    JobInfo,
    JobStatus,
    QueryRequest,
    QueryResponse,
)
from app.store import CollectionNotFound, DocumentStore, EmbeddingModelMismatch, create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    store: DocumentStore
    jobs: JobRegistry
    rag: RagPipeline


def build_state(settings: Settings, store: DocumentStore | None = None, llm: LLMClient | None = None) -> AppState:
    if store is None:
        store = DocumentStore(
            create_client(settings.chroma_persist_dir),
            create_embedder(settings.embedding_model),
            settings.embedding_batch_size,
        )
    window = store.embedder.max_input_tokens
    if window and settings.chunk_size > window:
        logger.warning(
            "CHUNK_SIZE=%d exceeds %s's input window of %d tokens; the end of each chunk "
            "will not be embedded and retrieval will miss it",
            settings.chunk_size,
            settings.embedding_model,
            window,
        )
    llm = llm or create_llm(settings)
    rag = RagPipeline(create_retriever(settings.retrieval_mode, store), llm, settings)
    return AppState(settings=settings, store=store, jobs=JobRegistry(), rag=rag)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set app.state.rag_state before startup to inject fakes.
    if not hasattr(app.state, "rag_state"):
        app.state.rag_state = build_state(get_settings())
    yield


app = FastAPI(title="RAG Generator", lifespan=lifespan)


def state(request: Request) -> AppState:
    return request.app.state.rag_state


def _require_collection(st: AppState, collection_id: str) -> None:
    if not st.store.exists(collection_id):
        raise HTTPException(status_code=404, detail=f"collection {collection_id!r} not found")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/collections", response_model=CollectionCreated, status_code=201)
def create_collection(body: CollectionCreate, st: AppState = Depends(state)) -> CollectionCreated:
    collection_id = st.store.create_collection(body.name)
    return CollectionCreated(collection_id=collection_id, name=body.name)


@app.get("/collections", response_model=list[CollectionInfo])
def list_collections(st: AppState = Depends(state)) -> list[CollectionInfo]:
    return [CollectionInfo(**c) for c in st.store.list_collections()]


def _run_ingestion(st: AppState, job_id: str, collection_id: str, files: list[tuple[str, bytes]]) -> None:
    st.jobs.set_status(job_id, JobStatus.running)
    any_failed = False
    for i, (filename, data) in enumerate(files):
        try:
            result = ingest_file(
                st.store, collection_id, filename, data, st.settings.chunk_size, st.settings.chunk_overlap
            )
        except Exception as exc:  # one bad file must not stop the rest
            logger.exception("ingestion failed for %s", filename)
            result = FileResult(filename=filename, status="failed", error=str(exc))
        any_failed |= result.status == "failed"
        st.jobs.set_file_result(job_id, i, result)
    st.jobs.set_status(job_id, JobStatus.failed if any_failed else JobStatus.done)


@app.post("/collections/{collection_id}/documents", response_model=JobCreated, status_code=202)
async def upload_documents(
    collection_id: str,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    st: AppState = Depends(state),
) -> JobCreated:
    _require_collection(st, collection_id)
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    max_bytes = st.settings.max_upload_mb * 1024 * 1024
    payload: list[tuple[str, bytes]] = []
    for f in files:
        filename = f.filename or "unnamed"
        try:
            check_supported(filename)
        except UnsupportedFileType as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Read one byte past the limit so oversized files are caught without reading them fully.
        data = await f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"{filename} exceeds the {st.settings.max_upload_mb} MB limit"
            )
        payload.append((filename, data))

    job_id = st.jobs.create(collection_id, [name for name, _ in payload])
    background.add_task(_run_ingestion, st, job_id, collection_id, payload)
    return JobCreated(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str, st: AppState = Depends(state)) -> JobInfo:
    job = st.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return job


def _query_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, CollectionNotFound):
        return HTTPException(status_code=404, detail="collection not found")
    if isinstance(exc, EmbeddingModelMismatch):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, LLMError):
        return HTTPException(status_code=502, detail=str(exc))
    raise exc


@app.post("/collections/{collection_id}/query", response_model=QueryResponse)
def query(collection_id: str, body: QueryRequest, st: AppState = Depends(state)) -> QueryResponse:
    _require_collection(st, collection_id)
    try:
        return st.rag.answer(collection_id, body.question, body.history)
    except (CollectionNotFound, EmbeddingModelMismatch, LLMError) as exc:
        raise _query_errors(exc) from exc


@app.post("/collections/{collection_id}/query/stream")
def query_stream(collection_id: str, body: QueryRequest, st: AppState = Depends(state)) -> StreamingResponse:
    """Same as /query but streams newline-delimited JSON events: sources, token..., done."""
    _require_collection(st, collection_id)
    events = st.rag.answer_events(collection_id, body.question, body.history)
    try:
        # Run retrieval eagerly so setup errors become proper HTTP status codes.
        first = next(events)
    except (CollectionNotFound, EmbeddingModelMismatch, LLMError) as exc:
        raise _query_errors(exc) from exc

    def ndjson() -> Iterator[str]:
        yield json.dumps(first) + "\n"
        try:
            for event in events:
                yield json.dumps(event) + "\n"
        except LLMError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")
