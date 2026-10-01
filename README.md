# RAG Generator

Upload documents at runtime, build a retrieval index over them, and ask questions
that get answered only from those documents, with inline citations. Works with any
document set without code changes.

- **Backend:** FastAPI, Chroma (one collection per document set), local
  `all-MiniLM-L6-v2` embeddings, Claude for generation
- **Frontend:** Streamlit, talking to the backend over HTTP only
- **Formats:** PDF (with page numbers), DOCX, TXT, MD

```
Upload ──> Parse ──> Chunk (+ metadata) ──> Embed ──> Chroma collection
                                                          │
Question ──> Rewrite follow-ups ──> Embed ──> Retrieve top-k ┘
                                                │
                       best score < threshold? ─┴─ yes ──> "not found" (no LLM call)
                                                │ no
                                  numbered context ──> LLM ──> answer + [file p.N] citations
```

## Quick start

Requires Python 3.11+ and an Anthropic API key.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then set ANTHROPIC_API_KEY
```

Run the API and the UI in two terminals:

```bash
uvicorn app.main:app --reload                 # http://localhost:8000/docs
streamlit run ui/streamlit_app.py             # http://localhost:8501
```

The first ingestion downloads the ONNX embedding model (about 80 MB) into `~/.cache/chroma`.

### Docker Compose

```bash
cp .env.example .env               # set ANTHROPIC_API_KEY
docker compose up --build
```

The UI is at http://localhost:8501 and the API at http://localhost:8000. The index
persists in the `chroma_data` volume.

## Using it

1. In the sidebar, create a document set and upload files.
2. Watch the ingestion job until every file shows `ingested` (or `skipped` for
   duplicates, `failed` with a reason).
3. Ask questions. Answers stream in, cite sources as `[filename p.N]`, and list the
   retrieved passages in a **Sources** expander. If the documents don't contain the
   answer, the reply is shown as a "Not found in the documents" warning, not as a
   normal answer.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/collections` | Create a document set: `{"name": "..."}` returns `collection_id` |
| `GET` | `/collections` | List sets with document and chunk counts |
| `POST` | `/collections/{id}/documents` | Multipart upload (`files`, repeatable). Returns `job_id` (202) |
| `GET` | `/jobs/{job_id}` | `pending` / `running` / `done` / `failed`, with per-file results |
| `POST` | `/collections/{id}/query` | `{question, history?}` returns `{answer, sources[], grounded, standalone_question}` |
| `POST` | `/collections/{id}/query/stream` | Same, as NDJSON events: `sources`, `token`..., `done` (or `error`) |

Unsupported file types return 400 and files over `MAX_UPLOAD_MB` return 413, both
before anything is ingested. Every query is scoped to exactly one collection.

```bash
CID=$(curl -s -X POST localhost:8000/collections -H 'content-type: application/json' \
  -d '{"name":"handbook"}' | jq -r .collection_id)
curl -s -F files=@eval/sample_docs/employee_handbook.pdf localhost:8000/collections/$CID/documents
curl -s -X POST localhost:8000/collections/$CID/query -H 'content-type: application/json' \
  -d '{"question":"How many PTO days do employees get?"}' | jq
```

## Configuration

Everything comes from environment variables or `.env` (see [.env.example](.env.example)).

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | From the environment or `.env`; held as a secret and never logged. If unset, the SDK falls back to `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any other name loads via `sentence-transformers` (install it separately) |
| `LLM_PROVIDER` | `anthropic` | Only provider implemented; see `app/llm.py` to add one |
| `LLM_MODEL` | `claude-opus-5` | |
| `LLM_REFUSAL_FALLBACK` | `true` | Server-side fallback if the model declines; disable on platforms that reject it |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `250` / `40` | Approximate tokens; see trade-offs below |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MIN_RELEVANCE_SCORE` | `0.25` | Cosine similarity; below this the app refuses without calling the LLM |
| `RETRIEVAL_MODE` | `dense` | `hybrid` adds BM25 with reciprocal rank fusion |
| `CHROMA_PERSIST_DIR` | (unset) | Unset means in-memory Chroma |
| `MAX_UPLOAD_MB` | `25` | Per file |
| `API_URL` | `http://localhost:8000` | Used by the UI and eval script |

## Grounding rules

These are enforced in [app/rag.py](app/rag.py) and covered by tests:

1. If the best retrieved chunk scores below `MIN_RELEVANCE_SCORE`, the response is
   `grounded: false` with a fixed message, and **the LLM is not called**.
2. The system prompt allows only facts from the context, requires inline
   `[filename p.N]` citations, and gives an exact sentence to use when the context
   is insufficient. If the model replies with that sentence, the API also returns
   `grounded: false`.
3. Context passages are numbered and labelled with filename and page.
4. Retrieved sources are always returned, even on refusal.
5. Follow-ups are rewritten into a standalone query from recent history before
   retrieval. The answer is generated from retrieved context only; history never
   reaches the answering prompt.

## Evaluation

[eval/run_eval.py](eval/run_eval.py) runs [eval/questions.yaml](eval/questions.yaml)
(12 answerable and 3 unanswerable questions) against a running API, using sample
docs for a fictional company so the model can't answer from memory.

```bash
python eval/run_eval.py                 # builds a fresh collection from eval/sample_docs
python eval/run_eval.py --no-judge      # skip the LLM-as-judge step
```

It reports:

- **Retrieval hit rate:** the expected file (and page, where given) is in the top-k
- **Faithfulness:** LLM-as-judge on whether each grounded answer is supported by the retrieved context
- **Refusal correctness:** unanswerable questions come back `grounded: false`
- **Expected facts:** key strings (such as "45" days) appear in the answer

Detailed per-question results are written to `eval/results/latest.json`.

## Design trade-offs

**In-memory vs persistent Chroma.** With `CHROMA_PERSIST_DIR` unset, Chroma runs in
memory: no setup and nothing to clean up, which suits tests and demos, but every
restart loses all collections. Setting it writes to disk (Docker Compose does this
by default). Ingestion job status is held in memory either way, so it is lost on
restart and doesn't work across multiple API workers. A real deployment would move
jobs to Redis or a database table.

**Chunk sizing.** The brief suggested about 800 tokens with 100 overlap, but
`all-MiniLM-L6-v2` only reads the first 256 word-pieces of its input and silently
drops the rest. At 800 tokens, facts near the end of a chunk were never embedded.
On the eval set, "How fast does the Ember stove boil a litre of water?" scored
0.065 because that paragraph sat at the end of a long chunk. At 250/40, every
answerable eval question retrieves its expected source as the top hit. The app logs
a warning at startup if `CHUNK_SIZE` exceeds the embedding model's window. If you
switch to a long-context embedding model, larger chunks give the LLM more
surrounding context per hit. Sizes are estimated at 4 characters per token rather
than using a tokenizer: the estimate only needs to be roughly right, and the
embedding model's tokenizer differs from the LLM's anyway.

**Threshold choice.** `MIN_RELEVANCE_SCORE=0.25` comes from the best-chunk scores on
the eval set with MiniLM at 250-token chunks:

| | Best-chunk cosine scores |
|---|---|
| Answerable (12) | 0.27 to 0.81 |
| Unanswerable (3) | 0.24, 0.28, 0.40 |

The two ranges overlap. "Does Halcyon ship to Canada?" (0.40) shares vocabulary with
the returns policy, so it outscores real questions. No threshold separates them
cleanly, so it is a coarse first filter: it catches clearly off-topic questions
cheaply and without an LLM call, and is set low enough not to refuse real
questions. The strict prompt (rule 2) handles the rest. Raising the threshold trades
false refusals for fewer LLM calls on off-topic questions. Scores depend on the
embedding model, so re-check this value if you change `EMBEDDING_MODEL`.

**Hybrid retrieval.** `RETRIEVAL_MODE=hybrid` merges BM25 and dense rankings with
reciprocal rank fusion. It helps with exact tokens like part numbers that small
embedding models blur together. RRF only picks and orders the chunks: each returned
chunk keeps its dense cosine score, so the threshold means the same thing in both
modes. On the small eval set both modes hit 12/12, so dense stays the default.

**Embeddings run locally.** The ONNX MiniLM model needs no API key or PyTorch and is
fast on CPU, but its quality is modest. The embedder sits behind a small interface
([app/embeddings.py](app/embeddings.py)), and the collection records which model
built it. Querying with a different `EMBEDDING_MODEL` returns 409 instead of
silently giving bad results.

## Project layout

```
app/
  main.py            FastAPI routes and app wiring
  config.py          Settings (pydantic-settings)
  schemas.py         Request/response models
  ingest/
    parsers.py       Extension dispatcher: PDF / DOCX / TXT / MD to pages
    chunker.py       Recursive splitter with overlap
    pipeline.py      Hash, dedupe, parse, chunk, embed, store
  embeddings.py      Embedding wrapper
  store.py           Chroma wrapper: collections, add, query
  retrieval.py       Dense and hybrid (BM25 + RRF) retrievers
  rag.py             Rewrite, retrieve, threshold, prompt, generate
  llm.py             LLM wrapper (Anthropic)
  jobs.py            In-memory ingestion job registry
ui/streamlit_app.py  Chat UI
eval/                Question set, sample docs, eval runner
tests/               pytest suite
```

## Tests

```bash
pytest
```

The tests use a deterministic fake embedder and a fake LLM, so they run offline in
a couple of seconds. They cover the parser dispatcher, the chunker, the threshold
refusal path (asserting the LLM is never called), collection isolation (store and
API level), dedup, embedding batching, the embedding-model mismatch check, hybrid
retrieval, and the HTTP endpoints including streaming.
