# CLAUDE.md

## Project

RAG Generator: users upload documents at runtime, the app builds a retrieval index over them, and users ask questions that get answers grounded in those documents. It must work with different document sets without code changes.

Time budget is about 8 hours. Favor simple, working, well-explained code over extra features.

## Scope

In scope:
- Upload documents (PDF, DOCX, TXT, MD) at runtime
- One Chroma collection per document set
- Question answering with citations
- A refusal when the documents don't contain the answer
- A small eval script

Out of scope for now. Do not add these unless asked:
- Tool calling or agent loops. The chat path is a plain retrieve-then-generate pipeline.
- Auth, multi-tenancy, user accounts
- Re-ranking models, fine-tuning, OCR for scanned PDFs

## Stack

- Backend: Python, FastAPI
- Frontend: Streamlit, which talks to the backend over HTTP only. It does not import backend modules.
- Vector store: Chroma. Use the persistent client when `chroma_persist_dir` is set, otherwise the in-memory client.
- Parsing: PyMuPDF for PDFs, python-docx for DOCX, plain read for TXT and MD
- Config: a plain `Settings` model in `app/config.py` for all settings; pydantic-settings `Secrets` reads only API keys from env or `.env`

## Architecture

```
Upload ──> Parse ──> Chunk (+ metadata) ──> Embed ──> Chroma collection
                                                          │
Question ──> Embed query (same model) ──> Retrieve top-k ─┘
                │
                └──> Score threshold check ──> Build prompt ──> LLM ──> Answer + citations
```

Ingestion and query must use the same embedding model. Store the model name in the collection metadata and reject queries if the configured model doesn't match.

## Suggested layout

```
app/
  main.py            # FastAPI app and routes
  config.py          # Settings (pydantic-settings)
  ingest/
    parsers.py       # Extension-based dispatcher; returns text with page numbers
    chunker.py       # Recursive splitter with overlap
  store.py           # Chroma client wrapper: collections, add, query
  rag.py             # Retrieve, threshold, prompt, generate
  llm.py             # Thin LLM client wrapper (provider set via config)
  schemas.py         # Pydantic request/response models
ui/
  streamlit_app.py
eval/
  questions.yaml     # 10-15 question/answer pairs over a sample doc set
  run_eval.py
tests/
docker-compose.yml
README.md
```

## API

- `POST /collections`: create a document set. Returns `collection_id`.
- `GET /collections`: list document sets with document counts.
- `POST /collections/{collection_id}/documents`: upload one or more files. Runs ingestion as a background task and returns `job_id`.
- `GET /jobs/{job_id}`: ingestion status (pending, running, done, failed) and per-file errors.
- `POST /collections/{collection_id}/query`: body `{question, history?}`. Returns `{answer, sources[], grounded: bool}`.

Every query is scoped to exactly one collection. Never search across collections.

## Ingestion rules

- Dispatch parsers by file extension. Reject unsupported types with a clear 400 error.
- Keep page numbers from PDFs so citations can point to a page.
- Chunking: recursive split on paragraphs, then sentences. Defaults are about 800 tokens with about 100 overlap, both configurable.
- Chunk metadata: `doc_id`, `filename`, `page`, `chunk_index`, `content_hash`. Chroma metadata values must be str, int, float, or bool. No lists, no None.
- Hash file contents with SHA-256 and skip files already in the collection.
- Batch embedding calls. Do not embed one chunk per request.

## Grounding rules

These are the core of the project. Keep them strict.

1. If the best retrieved chunk is below `MIN_RELEVANCE_SCORE`, return `grounded: false` with a fixed "not found in the documents" message, and do not call the LLM.
2. The system prompt tells the model to answer only from the provided context, to say so when the context is insufficient, and to cite sources inline as `[filename p.N]`.
3. Number each context chunk in the prompt with its filename and page so the model can cite it.
4. Always return the retrieved sources to the client, even when the answer cites only some of them.
5. For follow-up questions, rewrite the question into a standalone query using recent history before retrieval. Generate the answer from retrieved context, not from chat history.

## Config

Every setting has exactly one home, so no value is ever defined in two places:

- Settings (`embedding_model`, `llm_provider`, `llm_model`, `chunk_size`, `chunk_overlap`, `top_k`, `min_relevance_score`, `chroma_persist_dir`, `max_upload_mb`, and the rest) live only in `app/config.py` as defaults. They are not read from env vars or `.env`; a setting name found there is ignored with a startup warning.
- `.env` holds secrets only (API keys). Never add settings to `.env` or `.env.example`.

Do not hardcode any of these in application code. API keys come from env only and are never logged.

## Streamlit UI

- Sidebar: pick or create a collection, upload files, and show ingestion job status.
- Main area: chat. Stream the answer. Show sources in an expander under each reply.
- When `grounded` is false, display it clearly instead of showing it as a normal answer.

## Eval

`eval/run_eval.py` runs `questions.yaml` against a sample collection and reports:
- Retrieval hit rate: whether the expected source appears in the top-k
- Faithfulness: LLM-as-judge on whether the answer is supported by the retrieved context
- Refusal correctness on a few deliberately unanswerable questions

## Stretch goals (only after the core works)

- Hybrid retrieval: BM25 (`rank_bm25`) plus dense, merged with reciprocal rank fusion
- Streaming responses end to end
- Docker Compose for both services

## Conventions

- Type hints everywhere. Pydantic models for all request and response bodies.
- Keep the LLM and embedding clients behind small wrappers so providers can be swapped by config.
- Log ingestion and query timings at INFO level.
- Tests: pytest. Cover the parser dispatcher, the chunker, the threshold refusal path, and collection isolation.
- Update the README with run instructions and design trade-offs (in-memory vs persistent Chroma, chunk sizing, threshold choice) as features land.

## Time plan

| Block | Time |
|---|---|
| Parsing, chunking, embedding, Chroma collections | 2 h |
| Retrieval, prompting, citations, refusal threshold | 1.5 h |
| FastAPI endpoints and background ingestion | 1 h |
| Streamlit UI | 1.5 h |
| Eval script, README, Docker Compose | 1.5 h |
| Buffer | 0.5 h |