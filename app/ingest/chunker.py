"""Recursive text splitter with overlap.

Sizes are measured in approximate tokens (about 4 characters per token for
English). That avoids pulling in a tokenizer that may not match the embedding
model anyway; chunk size only needs to be roughly right.

Splitting order: paragraphs, then lines, then sentences, then words. Pieces are
greedily packed into chunks up to `chunk_size`; each new chunk starts with the
tail of the previous one (up to `chunk_overlap`) so context isn't lost at the
boundary.
"""

import re
from dataclasses import dataclass

from app.ingest.parsers import Page

CHARS_PER_TOKEN = 4

# Each separator is tried in order; a piece still too large moves to the next one.
_SEPARATORS: list[re.Pattern[str]] = [
    re.compile(r"\n\s*\n"),  # paragraphs
    re.compile(r"\n"),  # lines
    re.compile(r"(?<=[.!?])\s+"),  # sentences
    re.compile(r"\s+"),  # words
]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    chunk_index: int  # position within the document


def _split_pieces(text: str, max_chars: int, level: int = 0) -> list[str]:
    """Split `text` into pieces no longer than `max_chars`, using coarse separators first."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if level >= len(_SEPARATORS):
        # No separator left (e.g. one enormous "word"): hard cut.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    pieces: list[str] = []
    for part in _SEPARATORS[level].split(text):
        pieces.extend(_split_pieces(part, max_chars, level + 1))
    return pieces


def _overlap_tail(pieces: list[str], max_chars: int) -> list[str]:
    """Return the trailing pieces whose combined length fits in `max_chars`."""
    tail: list[str] = []
    size = 0
    for piece in reversed(pieces):
        if size + len(piece) > max_chars:
            break
        tail.insert(0, piece)
        size += len(piece) + 1
    return tail


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one block of text into overlapping chunks of roughly `chunk_size` tokens."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    max_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in _split_pieces(text, max_chars):
        if current and current_len + len(piece) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_chars)
            current_len = sum(len(p) + 1 for p in current)
            # If overlap + piece would still overflow, drop the overlap.
            if current_len + len(piece) > max_chars:
                current, current_len = [], 0
        current.append(piece)
        current_len += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one page for citations."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=text, page=page.page, chunk_index=len(chunks)))
    return chunks
