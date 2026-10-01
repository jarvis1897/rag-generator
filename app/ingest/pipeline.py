"""Ingest one file into a collection: hash, dedupe, parse, chunk, embed, store."""

import logging
import time

from app.ingest.chunker import chunk_pages
from app.ingest.parsers import parse_file
from app.schemas import FileResult
from app.store import DocumentStore, sha256_hex

logger = logging.getLogger(__name__)


def ingest_file(
    store: DocumentStore,
    collection_id: str,
    filename: str,
    data: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> FileResult:
    start = time.perf_counter()
    content_hash = sha256_hex(data)
    if store.has_document(collection_id, content_hash):
        logger.info("skipped %s: already in %s", filename, collection_id)
        return FileResult(filename=filename, status="skipped", error="already in this collection")

    pages = parse_file(filename, data)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    added = store.add_chunks(collection_id, filename, content_hash, chunks)
    logger.info(
        "ingested %s into %s: %d pages, %d chunks in %.2fs",
        filename,
        collection_id,
        len(pages),
        added,
        time.perf_counter() - start,
    )
    return FileResult(filename=filename, status="ingested", chunks=added)
