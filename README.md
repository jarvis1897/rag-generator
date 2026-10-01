# RAG Generator

Upload documents at runtime, build a retrieval index over them, and ask questions
that get answered only from those documents, with inline citations. Works with any
document set without code changes.

- **Backend:** FastAPI, Chroma (one collection per document set), Voyage AI
  `voyage-4-large` embeddings (2048 dims) and `rerank-2.5` reranking, Claude for generation.
  Local models (bge-m3, an ONNX cross-encoder) remain available as alternatives.
- **Frontend:** Streamlit, talking to the backend over HTTP only
- **Formats:** PDF (with page numbers), DOCX, TXT, MD

```
Upload ──> Parse ──> Chunk (+ metadata) ──> Embed (Voyage) ──> Chroma collection
                                                                    │
Question ──> Rewrite follow-ups ──> Embed ──> Hybrid retrieve 15 candidates (BM25 + dense)
                                                │
                       best score < threshold? ─┴─ yes ──> "not found" (no LLM call)
                                                │ no
                          rerank (Voyage) ──> top 5 ──> numbered context ──> Claude
                                                              ──> answer + [file p.N] citations
```

## Quick start

You need:

- Python 3.11+ (developed on 3.12)
- An Anthropic API key (https://console.anthropic.com)
- A Voyage AI API key (https://dash.voyageai.com). Add a payment method there: without one,
  the free-tier rate limits can throttle large ingests even though the free tokens cover the cost.

**1. Install**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

**2. Add your keys.** Copy the template and fill in both keys. `.env` holds secrets only;
every other setting lives in [app/config.py](app/config.py).

```bash
cp .env.example .env               # Windows PowerShell: Copy-Item .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
```

**3. Run the API and the UI** in two terminals, from the project root:

```bash
uvicorn app.main:app --reload                 # API: http://localhost:8000/docs
streamlit run ui/streamlit_app.py             # UI:  http://localhost:8501
```

The UI finds the API at `API_URL` (default `http://localhost:8000`). If the API runs
elsewhere, set it before starting Streamlit, for example `API_URL=http://localhost:9000`.

**4. Check it works** (optional): `pytest` runs the offline test suite in about 10 seconds,
and `python eval/run_eval.py` runs the eval against the running API (it makes real API calls).

With the default Voyage models nothing is downloaded. If you switch to the local models,
the first startup downloads them into `~/.cache/rag-generator/models` (bge-m3 is about 2.3 GB).

### Docker Compose

```bash
cp .env.example .env               # set ANTHROPIC_API_KEY and VOYAGE_API_KEY
docker compose up --build          # first build takes a minute or two
```

The UI is at http://localhost:8501 and the API at http://localhost:8000; stop any local
uvicorn or Streamlit first, since they use the same ports. The UI waits for the API's
health check before starting. The index persists in the `chroma_data` volume across
restarts and rebuilds, and local models (if you switch to them) cache in `model_cache`.
Images are built from your working copy, so edits to `app/config.py` take effect on the
next `--build`. `docker compose down` stops everything and keeps the data;
`docker compose down -v` also deletes it.

## Using it

1. In the sidebar, create a document set and upload files.
2. Watch the ingestion progress bar in the sidebar. It shows overall progress, an
   estimated time remaining, and per-file chunk counts, and refreshes on its own
   without interrupting the chat. Each file ends as `ingested`, `skipped` (duplicate)
   or `failed` with a reason.
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
| `GET` | `/jobs/{job_id}` | `pending` / `running` / `done` / `failed`, overall `progress` (0 to 1), `elapsed_seconds`, `eta_seconds`, and per-file status and `chunks_done`/`chunks` |
| `POST` | `/collections/{id}/query` | `{question, history?}` returns `{answer, sources[], grounded, standalone_question}` |
| `POST` | `/collections/{id}/query/stream` | Same, as NDJSON events: `sources`, `token`..., `done` (or `error`) |

Unsupported file types return 400 and files over `max_upload_mb` return 413, both
before anything is ingested. Every query is scoped to exactly one collection.

```bash
CID=$(curl -s -X POST localhost:8000/collections -H 'content-type: application/json' \
  -d '{"name":"handbook"}' | jq -r .collection_id)
curl -s -F files=@eval/sample_docs/employee_handbook.pdf localhost:8000/collections/$CID/documents
curl -s -X POST localhost:8000/collections/$CID/query -H 'content-type: application/json' \
  -d '{"question":"How many PTO days do employees get?"}' | jq
```

## Configuration

Each value has exactly one home, so nothing can be set in two places that disagree:

- **Settings** live only in [app/config.py](app/config.py), as defaults in the `Settings`
  class. Edit that file to change them. They are not read from environment variables or
  `.env`; if a setting name appears there, it is ignored and a warning is logged at startup.
- **Secrets** (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`) live only in `.env` or the
  environment, never in code or logs. If unset, the Anthropic SDK falls back to `ANTHROPIC_AUTH_TOKEN` or an
  `ant auth login` profile.

| Setting | Default | Notes |
|---|---|---|
| `embedding_model` | `voyage-4-large` | Any `voyage-*` model, or local `BAAI/bge-m3` / `all-MiniLM-L6-v2`; other names load via `sentence-transformers`. Changing it requires re-ingesting |
| `embedding_dimensions` | `2048` | Voyage only: 256, 512, 1024 or 2048. Changing it requires re-ingesting |
| `embedding_batch_size` | `None` | Chunks per embedding call; `None` uses the embedder's default (256 for Voyage, 8 for bge-m3) |
| `llm_provider` / `llm_model` | `anthropic` / `claude-opus-5` | Only Anthropic is implemented; see `app/llm.py` to add one |
| `llm_refusal_fallback` | `True` | Server-side fallback if the model declines; disable on platforms that reject it |
| `chunk_size` / `chunk_overlap` | `512` / `80` | Approximate tokens; see trade-offs below |
| `top_k` | `5` | Chunks sent to the LLM per question |
| `retrieval_candidates` | `15` | Chunks retrieved for the reranker to choose from (at least `top_k`) |
| `reranker_model` | `rerank-2.5` | Voyage reranker (32k-token input), `rerank-2.5-lite`, or a local ONNX cross-encoder such as `Xenova/ms-marco-MiniLM-L-6-v2`; `None` disables reranking |
| `min_relevance_score` | `0.25` | Cosine similarity; below this the app refuses without calling the LLM. Calibrated for voyage-4-large at 2048 dims (use 0.42 for bge-m3) |
| `retrieval_mode` | `hybrid` | BM25 + dense with reciprocal rank fusion; `dense` uses embeddings only |
| `chroma_persist_dir` | `./chroma_data` | `None` means in-memory Chroma |
| `max_upload_mb` | `25` | Per file |
| `model_cache_dir` | `~/.cache/rag-generator/models` | Downloaded models |

The Streamlit UI and the eval script are separate processes that don't import the
backend; they find the API at `API_URL` (default `http://localhost:8000`).

## Grounding rules

These are enforced in [app/rag.py](app/rag.py) and covered by tests:

1. If the best retrieved chunk scores below `min_relevance_score`, the response is
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

[eval/run_eval.py](eval/run_eval.py) runs a question set against a running API. It
uploads every file in the `--docs` folder into a fresh collection, so each eval set
keeps its documents in its own folder:

| Question set | Documents | What it tests |
|---|---|---|
| [eval/questions.yaml](eval/questions.yaml) (default) | `eval/sample_docs/` | 12 answerable and 3 unanswerable questions over docs for a fictional company, so the model can't answer from memory |
| [eval/pg18_questions.yaml](eval/pg18_questions.yaml) | `eval/pg18_docs/` | 6 answerable and 3 unanswerable questions over *The Federalist Papers* (456-page PDF, about 780 chunks): page-level retrieval in a long document |

```bash
python eval/run_eval.py                 # default set: builds a fresh collection from eval/sample_docs
python eval/run_eval.py --no-judge      # skip the LLM-as-judge step
python eval/run_eval.py --questions eval/pg18_questions.yaml --docs eval/pg18_docs --out eval/results/pg18.json
python eval/run_eval.py --collection-id <id> ...   # reuse an already-ingested collection
```

It reports:

- **Retrieval hit rate:** the expected file (and page, where given) is in the top-k
- **Faithfulness:** LLM-as-judge on whether each grounded answer is supported by the retrieved context
- **Refusal correctness:** unanswerable questions come back `grounded: false`
- **Context recall:** the answer's key facts appear in the sources sent to the LLM (missed ground truth shows up here)
- **MRR:** mean reciprocal rank of the expected source among those sources
- **Expected facts:** key strings (such as "45" days) appear in the answer

Detailed per-question results are written to `eval/results/latest.json` (or `--out`).
The script reads the API keys from `.env` for the judge, so run it from the project root.

Latest results (Voyage embeddings and reranking, hybrid retrieval, run against the
Docker Compose stack):

| Metric | Default set | Federalist Papers |
|---|---|---|
| Expected source in top-k / expected page | 12/12 / 4/4 | 6/6 / 6/6 |
| Context recall / MRR | 10/10 / 1.00 | 6/6 / 1.00 |
| Faithfulness (LLM judge) | 12/12 | 6/6 |
| Refusal correct on unanswerable | 3/3 | 3/3 |

Known gap: broad questions ("what arguments does Federalist No. 78 make?") span more
text than the 5 chunks sent to the LLM, so answers can be faithful but incomplete.
Raising `top_k` helps at the cost of longer prompts.

## Design trade-offs

**In-memory vs persistent Chroma.** By default Chroma writes to `./chroma_data`, so
collections survive restarts. Setting `chroma_persist_dir = None` runs it in memory:
no setup and nothing to clean up, which suits tests and demos, but every restart
loses all collections. Ingestion job status is held in memory either way, so it is lost on
restart and doesn't work across multiple API workers. A real deployment would move
jobs to Redis or a database table.

**Embedding model and chunk sizing.** The embedding model went through three versions:

| | all-MiniLM-L6-v2 | BAAI/bge-m3 | voyage-4-large (current) |
|---|---|---|---|
| Runs | Locally | Locally | Voyage API |
| Input window | 256 tokens | 8192 tokens | 32,000 tokens |
| Vector size | 384 | 1024 | 2048 (configurable) |
| Ingestion speed on this laptop's CPU | Fast | About 1.5 s per chunk (781 chunks: 19 min) | Seconds per document |

MiniLM's 256-token window capped chunks at about 250 tokens, too small for legal text
where a clause and its conditions, exceptions and defined terms need to stay together.
bge-m3 fixed that and retrieved better (MRR 1.00 vs 0.94 on the eval questions), but at
about 1.5 s per chunk on CPU it made ingestion far too slow. Quantizing it to int8 only
gained about 20%, and GPU inference couldn't be verified. voyage-4-large runs remotely,
so ingestion takes seconds, and it is Voyage's best general retrieval model. Vectors are
2048-dimensional, the largest it offers: about twice the storage of 1024, for a small
quality gain.

Chunks are 512 tokens with 80 overlap. With Voyage embeddings and reranking, both read
32k tokens, so chunks could be larger; the trade-off is less precise citations and
more text per prompt. The local ONNX reranker reads only 512 tokens, so keep chunks at
512 if you switch back to it. The app logs a warning at startup if `chunk_size`
exceeds the embedding model's window. Sizes are estimated at 4 characters per token
rather than with a tokenizer; the estimate only needs to be roughly right.

**Threshold choice.** `min_relevance_score = 0.25` comes from best-chunk cosine scores
on the sample docs with voyage-4-large (2048 dims) at 512-token chunks:

| Question type | voyage-4-large | bge-m3 (for comparison) |
|---|---|---|
| Answerable, including 10 paraphrases (22) | 0.306 to 0.589 | 0.467 to 0.725 |
| On-topic but unanswerable (3) | 0.217, 0.289, 0.296 | 0.360, 0.448, 0.551 |
| Clearly off-topic (5) | 0.066 to 0.170 | 0.285 to 0.383 |

Voyage separates the groups far better: with bge-m3, an unanswerable question
outscored real ones. 0.25 refuses every off-topic question without an LLM call (0.08
of margin) and keeps every answerable one (0.056 of margin). It also refuses one of
the three unanswerable questions; the other two (0.289, 0.296) are handled by the
strict prompt (rule 2). Raising the threshold to 0.30 would catch them too, but would
leave only 0.006 of margin before real questions get refused. Scores depend on the
embedding model and vector size, so re-check this whenever either changes, ideally on
your own documents: raising it trades false refusals for fewer LLM calls. With
bge-m3, use 0.42.

**Hybrid retrieval.** `retrieval_mode = "hybrid"` merges BM25 and dense rankings with
reciprocal rank fusion. RRF only picks and orders the chunks: each returned
chunk keeps its dense cosine score, so the threshold means the same thing in both
modes. It is the default because dense-only retrieval missed too much ground truth
in practice: keyword matches catch exact names, numbers, section references and
defined terms that embeddings blur together. With reranking on, hybrid fills the
15-candidate pool from both rankings (up to 60 chunks each before fusion), and the
reranker then picks the best 5. The cost is a BM25 index per collection held in memory, rebuilt
when documents are added. Set `retrieval_mode = "dense"` to go back.

**Reranking.** The embedding model scores question and passage separately, so it
can rank a passage that merely shares vocabulary above the one that answers the
question, or push the answer below the top 5. Retrieving 15 candidates and letting a
cross-encoder pick the best 5 widens the net for recall while keeping the prompt
short. The cross-encoder reads question and passage together, which is more
accurate but too slow to run over a whole collection. The default is Voyage's
`rerank-2.5`, which reads up to 32k tokens of query plus passage, so long chunks are
scored in full. The local ONNX alternative adds about 0.2 s per query on CPU but reads
only 512 tokens. The threshold still uses dense cosine over all 15 candidates, so it
means the same thing whichever reranker runs (local cross-encoder scores are
unbounded logits with no stable cutoff anyway). Sources carry both `score` (cosine)
and `rerank_score`. When reranking was first added (local cross-encoder, MiniLM
embeddings), recall@5 on the small sample set was already 20/20, but reranking moved
the answer chunk higher: MRR went from 0.93 to 0.97. Expect the recall gain on larger collections, where the answer can rank
below 5th. Measure it with `run_eval.py`'s context recall on your own documents.

**Ingestion progress.** Nearly all ingestion time is embedding, so progress is
reported after each embedding batch. A file's chunk count isn't known until it is
parsed, so overall progress weights files by size in bytes. The time remaining is a
linear extrapolation from elapsed time, which holds up because embedding speed is
roughly constant per chunk. It is hidden for the first 2% because early estimates
are noise. Embedding a 157-chunk contract with bge-m3 took about 70 s on CPU. With Voyage,
progress advances in steps of 256 chunks per request batch.

**Hosted vs local models.** Voyage (Anthropic's recommended embeddings provider;
Anthropic has no embedding model of its own) makes ingestion fast and needs no local
compute, but document text and questions are sent to Voyage's servers, it needs
internet access and a `VOYAGE_API_KEY`, and each question adds an embedding call and
a rerank call. It is paid per token, with free allowances (200M tokens for
voyage-4-large and rerank-2.5 at the time of writing); a 781-chunk document is about
400k tokens. Add a payment method to the Voyage account, because the limits without
one can throttle large ingests. For confidential material that must not leave the
machine, set `embedding_model = "BAAI/bge-m3"` and
`reranker_model = "Xenova/ms-marco-MiniLM-L-6-v2"` (and `embedding_dimensions = None`).
Both kinds of model sit behind small interfaces ([app/embeddings.py](app/embeddings.py),
[app/reranker.py](app/reranker.py), [app/voyage.py](app/voyage.py)). Each collection
records which model and vector size built it, and querying with a different
configuration returns 409 instead of silently giving bad results.

## Project layout

```
app/
  main.py            FastAPI routes and app wiring
  config.py          Settings (all defaults) and Secrets (API keys from .env)
  schemas.py         Request/response models
  ingest/
    parsers.py       Extension dispatcher: PDF / DOCX / TXT / MD to pages
    chunker.py       Recursive splitter with overlap
    pipeline.py      Hash, dedupe, parse, chunk, embed, store
  embeddings.py      Embedding wrapper
  store.py           Chroma wrapper: collections, add, query
  retrieval.py       Dense and hybrid (BM25 + RRF) retrievers
  reranker.py        Reranker factory and local ONNX cross-encoder
  voyage.py          Voyage AI embeddings and reranking
  errors.py          Provider error type
  hf.py              Model downloads from the Hugging Face Hub
  rag.py             Rewrite, retrieve, threshold, prompt, generate
  llm.py             LLM wrapper (Anthropic)
  jobs.py            In-memory ingestion job registry with progress and ETA
ui/streamlit_app.py  Chat UI (HTTP only, no backend imports)
eval/
  run_eval.py        Eval runner
  questions.yaml     Default question set, documents in sample_docs/
  pg18_questions.yaml  Federalist Papers set, documents in pg18_docs/
  make_sample_pdf.py Regenerates sample_docs/employee_handbook.pdf
tests/               pytest suite (offline: fake embedder, LLM, reranker, Voyage client)
Dockerfile.api, Dockerfile.ui, docker-compose.yml
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Startup fails with `VOYAGE_API_KEY is not set` | Add the key to `.env` in the project root and restart |
| 502 "rejected the API key" | The Anthropic or Voyage key in `.env` is wrong or revoked |
| 502 "Voyage rate limit hit" | Add a payment method to the Voyage account |
| 409 on query: embedding model mismatch | The collection was built with a different `embedding_model` or `embedding_dimensions`. Re-ingest into a new collection, or clear the index |
| Warning "ignoring X set in .env" | Settings belong in `app/config.py`; remove them from `.env` |
| Collections disappear on restart | `chroma_persist_dir` is `None` (in-memory); set it to `"./chroma_data"` |
| UI says it can't reach the API | Start uvicorn first, or set `API_URL` to where it runs |

**Clearing the index:** stop the API and delete the `chroma_data` folder (with Docker:
`docker compose down -v`). Both commands delete every collection.

## Tests

```bash
pytest
```

The tests use a deterministic fake embedder and a fake LLM, so they run offline in
a couple of seconds. They cover the parser dispatcher, the chunker, the threshold
refusal path (asserting the LLM is never called), reranking (candidates in, top-k
out), the Voyage clients (against a fake client: input types, request splitting,
error mapping), collection isolation (store and API level), dedup, embedding
batching, the embedding-model mismatch check, hybrid retrieval, ingestion progress,
the single-source config rules, and the HTTP endpoints including streaming.
