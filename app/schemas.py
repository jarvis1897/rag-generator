"""Pydantic request and response models for the HTTP API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class CollectionInfo(BaseModel):
    collection_id: str
    name: str
    document_count: int
    chunk_count: int
    embedding_model: str


class CollectionCreated(BaseModel):
    collection_id: str
    name: str


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class FileResult(BaseModel):
    filename: str
    status: Literal["pending", "ingested", "skipped", "failed"]
    chunks: int = 0
    error: str | None = None


class JobInfo(BaseModel):
    job_id: str
    collection_id: str
    status: JobStatus
    files: list[FileResult]


class JobCreated(BaseModel):
    job_id: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)


class Source(BaseModel):
    index: int  # the [n] number used in the prompt
    filename: str
    page: int
    chunk_index: int
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    standalone_question: str
