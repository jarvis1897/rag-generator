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
- Reranking (added on request): retrieve `retrieval_candidates`, rerank, send `top_k` to the LLM

Out of scope for now. Do not add these unless asked:
- Tool calling or agent loops. The chat path is a plain retrieve-then-generate pipeline.
- Auth, multi-tenancy, user accounts
- Fine-tuning, OCR for scanned PDFs

## Stack

- Backend: Python, FastAPI
- Frontend: Streamlit, which talks to the backend over HTTP only. It does not import backend modules.
- Vector store: Chroma. Use the persistent client when `chroma_persist_dir` is set, otherwise the in-memory client.
- Parsing: PyMuPDF for PDFs, python-docx for DOCX, plain read for TXT and MD
- Embeddings and reranking: Voyage AI (`voyage-4-large` at 2048 dims, `rerank-2.5`) by default, behind wrappers. Local alternatives (bge-m3, an ONNX cross-encoder) stay selectable for documents that must not leave the machine.
- LLM: Anthropic Claude (`claude-opus-5`)
- Config: a plain `Settings` model in `app/config.py` for all settings; pydantic-settings `Secrets` reads only API keys from env or `.env`

## Architecture

```
Upload ──> Parse ──> Chunk (+ metadata) ──> Embed (batched) ──> Chroma collection
                                                                      │
Question ──> Rewrite follow-up ──> Embed query (same model) ──> Hybrid retrieve (BM25 + dense, RRF)
                                                                      │  retrieval_candidates
                         Score threshold check (dense cosine) <───────┘
                           │ pass                    │ fail
                           ▼                         └──> grounded: false, no LLM call
                         Rerank ──> top_k ──> Build prompt ──> LLM ──> Answer + citations
```

Ingestion and query must use the same embedding model and vector size. Store the model name (with dimensions, e.g. `voyage-4-large@2048`) in the collection metadata and reject queries if the configured model doesn't match. The threshold always uses dense cosine, never reranker scores, so it means the same thing whichever reranker runs.

## Layout

```
app/
  main.py            # FastAPI app, routes, and wiring (build_state)
  config.py          # Settings (plain model, all defaults) + Secrets (API keys from env/.env)
  errors.py          # ProviderError for embedding/reranking provider failures
  ingest/
    parsers.py       # Extension-based dispatcher; returns text with page numbers
    chunker.py       # Recursive splitter with overlap
    pipeline.py      # Hash, dedupe, parse, chunk, embed, store one file (with progress callback)
  embeddings.py      # Embedder interface + local embedders (bge-m3, MiniLM) and factory
  voyage.py          # Voyage AI embedder and reranker (the defaults)
  reranker.py        # Reranker interface, local ONNX cross-encoder, and factory
  hf.py              # Hugging Face model downloads for the local models
  store.py           # Chroma client wrapper: collections, add, query
  retrieval.py       # Dense and hybrid (BM25 + RRF) retrievers
  rag.py             # Rewrite, retrieve, threshold, rerank, prompt, generate
  llm.py             # Thin LLM client wrapper (provider set via config)
  jobs.py            # In-memory ingestion job registry with progress and ETA
  schemas.py         # Pydantic request/response models
ui/
  streamlit_app.py   # Talks to the API over HTTP only
eval/
  run_eval.py        # Runs a question set: --questions, --docs, --collection-id, --out
  questions.yaml     # Default set over sample_docs/ (fictional company)
  sample_docs/       # Documents for questions.yaml only
  pg18_questions.yaml  # Federalist Papers set over pg18_docs/
  pg18_docs/         # Documents for pg18_questions.yaml only
  make_sample_pdf.py # Regenerates sample_docs/employee_handbook.pdf
tests/               # pytest; fakes for the embedder, LLM, reranker and Voyage client
Dockerfile.api, Dockerfile.ui, docker-compose.yml
README.md
```

Each eval set keeps its documents in its own folder, because run_eval.py uploads every file in `--docs`. Never add documents for one set to another set's folder.

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
- Chunking: recursive split on paragraphs, then sentences. Defaults are 512 tokens with 80 overlap (`chunk_size`, `chunk_overlap`); see the README for why not larger.
- Chunk metadata: `doc_id`, `filename`, `page`, `chunk_index`, `content_hash`. Chroma metadata values must be str, int, float, or bool. No lists, no None.
- Hash file contents with SHA-256 and skip files already in the collection.
- Batch embedding calls. Do not embed one chunk per request.

## Grounding rules

These are the core of the project. Keep them strict.

1. If the best retrieved chunk is below `min_relevance_score`, return `grounded: false` with a fixed "not found in the documents" message, and do not call the LLM.
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

All three are done:

- Hybrid retrieval: BM25 (`rank_bm25`) plus dense, merged with reciprocal rank fusion (the default)
- Streaming responses end to end (`/query/stream`, used by the UI)
- Docker Compose for both services (verified end to end)

## Future work: graph knowledge store and retrieval

Not started. The next major feature, for relational and multi-step questions that chunk retrieval handles poorly: "Which papers by Madison cite the Swiss confederacy?", "Who is bound by the clause that section 4 refers to?", "Compare how No. 10 and No. 51 treat faction." Their answer is spread across chunks that share no wording with the question, so dense, BM25 and reranking all miss some of the pieces.

Goal: alongside each Chroma collection, build a knowledge graph of entities (people, organizations, documents and sections, defined terms, dates, amounts) and the relations between them, and use it to pull in the related chunks a question needs before generation.

Design constraints, so it fits the rest of the system:

- **Scoped per collection.** One graph per document set, created and deleted with its Chroma collection. Never traverse across collections.
- **Every node and edge points back to source chunks** (`doc_id`, `chunk_index`, `filename`, `page`). The LLM still answers only from chunk text, never from bare graph facts, so the citation rules (`[filename p.N]`) and grounding rules 1 to 5 stay unchanged.
- **Retrieval stays deterministic, not an agent loop.** Find entities in the (standalone) question, expand a bounded number of hops in the graph (setting, e.g. `graph_max_hops = 2`), collect the linked chunks, and merge them with the hybrid candidates (RRF) before the threshold check and reranking. No tool calling, no model-driven multi-turn search; that remains out of scope.
- **The threshold keeps its meaning.** Graph-sourced chunks get their dense cosine score, like BM25-only hits do today, so `min_relevance_score` still gates the LLM call.
- **Extraction runs at ingestion, in the background job**, batched (never one LLM call per chunk), reported through the existing job progress, and deduplicated by `content_hash` like everything else. Entity resolution (merging "Madison", "James Madison", "Publius") is the hard part; record which method and model built the graph in the collection metadata, as with embeddings, and reject mismatches.
- **Behind a small interface with a config switch** (e.g. `graph_mode: Literal["off", "on"]` in `app/config.py`, default `"off"`), so the plain pipeline keeps working and the graph can be compared against it. Pick a store that is simple to run locally and in Docker Compose (an embedded graph or plain tables first; add a server like Neo4j only if the eval shows the need).
- **Measure before and after.** Add a multi-hop question set to `eval/` (its own `*_questions.yaml` and docs folder) whose answers need two or more chunks, and report context recall and faithfulness with the graph on and off. Ship it only if it beats the hybrid + rerank baseline without hurting the existing sets.

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