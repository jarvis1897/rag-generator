"""Streamlit chat UI. Talks to the FastAPI backend over HTTP only."""

import json
import os
import time
from collections.abc import Iterator

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 30

st.set_page_config(page_title="RAG Generator", page_icon="📚", layout="wide")


# --- API helpers ---


def api(method: str, path: str, **kwargs) -> requests.Response:
    return requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", TIMEOUT), **kwargs)


def error_detail(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text


def list_collections() -> list[dict]:
    resp = api("GET", "/collections")
    resp.raise_for_status()
    return resp.json()


def stream_query(collection_id: str, question: str, history: list[dict]) -> Iterator[dict]:
    with requests.post(
        f"{API_URL}/collections/{collection_id}/query/stream",
        json={"question": question, "history": history},
        stream=True,
        timeout=(10, 300),
    ) as resp:
        if resp.status_code != 200:
            yield {"type": "error", "detail": error_detail(resp)}
            return
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


# --- rendering ---


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['index']}] {s['filename']} p.{s['page']}** · score {s['score']:.3f}")
            st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["content"])
    elif not msg.get("grounded", True):
        st.warning(f"**Not found in the documents.** {msg['content']}", icon="🔎")
    else:
        st.markdown(msg["content"])
    render_sources(msg.get("sources", []))


# --- sidebar: collections, uploads, job status ---

if "messages" not in st.session_state:
    st.session_state.messages = {}  # collection_id -> list of chat messages
if "jobs" not in st.session_state:
    st.session_state.jobs = []  # job ids started in this session

with st.sidebar:
    st.header("Document sets")
    try:
        collections = list_collections()
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API_URL}: {exc}")
        st.stop()

    with st.form("new_collection", clear_on_submit=True):
        new_name = st.text_input("New document set", placeholder="e.g. HR policies")
        if st.form_submit_button("Create") and new_name.strip():
            resp = api("POST", "/collections", json={"name": new_name.strip()})
            if resp.ok:
                st.session_state.selected = resp.json()["collection_id"]
                st.rerun()
            st.error(error_detail(resp))

    if not collections:
        st.info("Create a document set to get started.")
        st.stop()

    ids = [c["collection_id"] for c in collections]
    by_id = {c["collection_id"]: c for c in collections}
    selected = st.session_state.get("selected")
    collection_id = st.selectbox(
        "Active document set",
        ids,
        index=ids.index(selected) if selected in ids else 0,
        format_func=lambda cid: f"{by_id[cid]['name']} ({by_id[cid]['document_count']} docs)",
    )
    st.session_state.selected = collection_id

    st.subheader("Upload")
    files = st.file_uploader(
        "PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Ingest", disabled=not files, width="stretch"):
        resp = api(
            "POST",
            f"/collections/{collection_id}/documents",
            files=[("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files],
            timeout=120,
        )
        if resp.ok:
            st.session_state.jobs.insert(0, resp.json()["job_id"])
        else:
            st.error(error_detail(resp))

    if st.session_state.jobs:
        st.subheader("Ingestion jobs")
        running = False
        for job_id in st.session_state.jobs[:5]:
            resp = api("GET", f"/jobs/{job_id}")
            if not resp.ok:
                continue
            job = resp.json()
            running |= job["status"] in ("pending", "running")
            icon = {"pending": "⏳", "running": "⚙️", "done": "✅", "failed": "⚠️"}[job["status"]]
            with st.expander(f"{icon} {len(job['files'])} file(s), {job['status']}", expanded=running):
                for f in job["files"]:
                    line = f"`{f['filename']}`: {f['status']}"
                    if f["chunks"]:
                        line += f" ({f['chunks']} chunks)"
                    if f["error"]:
                        line += f" · {f['error']}"
                    st.markdown(line)
        if running:
            time.sleep(1.5)
            st.rerun()


# --- main area: chat ---

st.title(by_id[collection_id]["name"])
st.caption(f"Answers come only from this document set · embeddings: {by_id[collection_id]['embedding_model']}")

messages: list[dict] = st.session_state.messages.setdefault(collection_id, [])
for msg in messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

if question := st.chat_input("Ask a question about these documents"):
    history = [
        {"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")
    ]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        reply: dict = {"role": "assistant", "content": "", "sources": [], "grounded": True}
        placeholder = st.empty()
        try:
            for event in stream_query(collection_id, question, history):
                if event["type"] == "sources":
                    reply["sources"] = event["sources"]
                elif event["type"] == "token":
                    reply["content"] += event["text"]
                    placeholder.markdown(reply["content"] + "▌")
                elif event["type"] == "done":
                    reply["content"] = event["answer"]
                    reply["grounded"] = event["grounded"]
                elif event["type"] == "error":
                    reply.update(content=event["detail"], error=True)
        except requests.RequestException as exc:
            reply.update(content=f"Request failed: {exc}", error=True)
        placeholder.empty()
        render_assistant(reply)
    messages.append(reply)
