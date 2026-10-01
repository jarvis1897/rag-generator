import io

import pytest

from app.ingest.chunker import CHARS_PER_TOKEN, chunk_pages, split_text
from app.ingest.parsers import Page, ParseError, UnsupportedFileType, parse_file


def make_pdf(pages: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_docx(paragraphs: list[str]) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


# --- parser dispatcher ---


def test_pdf_keeps_page_numbers() -> None:
    pages = parse_file("report.PDF", make_pdf(["first page text", "second page text"]))
    assert [p.page for p in pages] == [1, 2]
    assert "second page" in pages[1].text


def test_pdf_drops_blank_pages() -> None:
    pages = parse_file("r.pdf", make_pdf(["has text", "", "more text"]))
    assert [p.page for p in pages] == [1, 3]


def test_docx() -> None:
    pages = parse_file("notes.docx", make_docx(["Alpha paragraph.", "Beta paragraph."]))
    assert len(pages) == 1 and pages[0].page == 1
    assert "Alpha" in pages[0].text and "Beta" in pages[0].text


@pytest.mark.parametrize("name", ["a.txt", "b.md", "C.MD"])
def test_text_formats(name: str) -> None:
    pages = parse_file(name, "# Title\n\nhello wörld".encode())
    assert pages == [Page(page=1, text="# Title\n\nhello wörld")]


def test_text_strips_bom() -> None:
    assert parse_file("a.txt", "﻿hi".encode("utf-8"))[0].text == "hi"


@pytest.mark.parametrize("name", ["image.png", "sheet.xlsx", "noextension", "archive.tar.gz"])
def test_unsupported_type_rejected(name: str) -> None:
    with pytest.raises(UnsupportedFileType):
        parse_file(name, b"whatever")


def test_corrupt_pdf_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_file("bad.pdf", b"not a pdf")


def test_empty_text_file_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_file("empty.txt", b"   \n  ")


# --- chunker ---


def test_short_text_is_one_chunk() -> None:
    assert split_text("Just a sentence.", chunk_size=100, chunk_overlap=10) == ["Just a sentence."]


def test_chunks_respect_size() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 * CHARS_PER_TOKEN for c in chunks)


def test_chunks_overlap() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(200)]
    chunks = split_text(" ".join(sentences), chunk_size=50, chunk_overlap=15)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.split(". ")[-1]
        assert last_sentence.rstrip(".") in nxt  # tail of one chunk starts the next


def test_prefers_paragraph_boundaries() -> None:
    para_a = "A" * 300
    para_b = "B" * 300
    chunks = split_text(f"{para_a}\n\n{para_b}", chunk_size=100, chunk_overlap=0)
    assert chunks == [para_a, para_b]


def test_giant_word_is_hard_cut() -> None:
    chunks = split_text("x" * 1000, chunk_size=50, chunk_overlap=0)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == "x" * 1000


def test_overlap_must_be_smaller() -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, chunk_overlap=10)


def test_chunk_pages_tracks_page_and_index() -> None:
    pages = [Page(1, "one " * 300), Page(2, "two " * 300)]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert all(("one" in c.text) != ("two" in c.text) for c in chunks)  # no chunk spans pages
