# RAG Generator

Upload documents at runtime, build a retrieval index over them, and ask questions
that get answered only from those documents, with inline citations. Works with any
document set without code changes.

- **Backend:** FastAPI, Chroma (one collection per document set), local `BAAI/bge-m3`
  embeddings (8192-token window, 1024 dims), a cross-encoder reranker, Claude for generation
- **Frontend:** Streamlit, talking to the backend over HTTP only
- **Formats:** PDF (with page numbers), DOCX, TXT, MD

```
Upload ──> Parse ──> Chunk (+ metadata) ──> Embed ──> Chroma collection
                                                          │
Question ──> Rewrite follow-ups ──> Embed ──> Retrieve 15 candidates ┘
                                                │
                       best score < threshold? ─┴─ yes ──> "not found" (no LLM call)
                                                │ no
                         cross-encoder rerank ──> top 5 ──> numbered context ──> LLM
                                                              ──> answer + [file p.N] citations
```

## Quick start

Requires Python 3.11+ and an Anthropic API key.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then set ANTHROPIC_API_KEY (the only thing .env holds)
```

Run the API and the UI in two terminals:

```bash
uvicorn app.main:app --reload                 # http://localhost:8000/docs
streamlit run ui/streamlit_app.py             # http://localhost:8501
```

The first startup downloads the embedding model (bge-m3, about 2.3 GB) and the reranker
(about 90 MB) into `~/.cache/rag-generator/models`, so it takes a few minutes once. Later
starts load them in about 5 seconds.

### Docker Compose

```bash
cp .env.example .env               # set ANTHROPIC_API_KEY
docker compose up --build
```

The UI is at http://localhost:8501 and the API at http://localhost:8000. The index
persists in the `chroma_data` volume and the models in `model_cache`.

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
- **Secrets** (`ANTHROPIC_API_KEY`) live only in `.env` or the environment, never in code
  or logs. If unset, the Anthropic SDK falls back to `ANTHROPIC_AUTH_TOKEN` or an
  `ant auth login` profile.

| Setting | Default | Notes |
|---|---|---|
| `embedding_model` | `BAAI/bge-m3` | Also `all-MiniLM-L6-v2`; any other name loads via `sentence-transformers` (install separately). Changing it requires re-ingesting |
| `embedding_batch_size` | `8` | Chunks per embedding call; small because chunks are long |
| `llm_provider` / `llm_model` | `anthropic` / `claude-opus-5` | Only Anthropic is implemented; see `app/llm.py` to add one |
| `llm_refusal_fallback` | `True` | Server-side fallback if the model declines; disable on platforms that reject it |
| `chunk_size` / `chunk_overlap` | `512` / `80` | Approximate tokens; see trade-offs below |
| `top_k` | `5` | Chunks sent to the LLM per question |
| `retrieval_candidates` | `15` | Chunks retrieved for the reranker to choose from (at least `top_k`) |
| `reranker_model` | `Xenova/ms-marco-MiniLM-L-6-v2` | ONNX cross-encoder; `None` disables reranking |
| `min_relevance_score` | `0.42` | Cosine similarity; below this the app refuses without calling the LLM. Calibrated for bge-m3 |
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
- **Context recall:** the answer's key facts appear in the sources sent to the LLM (missed ground truth shows up here)
- **MRR:** mean reciprocal rank of the expected source among those sources
- **Expected facts:** key strings (such as "45" days) appear in the answer

Detailed per-question results are written to `eval/results/latest.json`.

## Design trade-offs

**In-memory vs persistent Chroma.** By default Chroma writes to `./chroma_data`, so
collections survive restarts. Setting `chroma_persist_dir = None` runs it in memory:
no setup and nothing to clean up, which suits tests and demos, but every restart
loses all collections. To clear the index, stop the API and delete `chroma_data`. Ingestion job status is held in memory either way, so it is lost on
restart and doesn't work across multiple API workers. A real deployment would move
jobs to Redis or a database table.

**Embedding model and chunk sizing.** The first version used `all-MiniLM-L6-v2`, which
reads only 256 word-pieces and silently drops the rest. That capped chunks at about 250
tokens: at 800, a fact at the end of a chunk was never embedded (an eval question about
the Ember stove scored 0.065 for that reason). That is too small for legal and similar
documents, where a clause and its conditions, exceptions and defined terms need to stay
in one chunk. The default is now `BAAI/bge-m3`:

| | all-MiniLM-L6-v2 | BAAI/bge-m3 |
|---|---|---|
| Input window | 256 tokens | 8192 tokens |
| Vector size | 384 | 1024 |
| Download | about 80 MB | about 2.3 GB |
| Chunk-level recall@5 / MRR (20 eval + paraphrased questions) | 20/20 / 0.94 | 20/20 / 1.00 |

Chunks are 512 tokens with 80 overlap. bge-m3 could take far more, but the reranker
reads at most 512 tokens of question plus passage, so larger chunks would be partly
invisible to it. Larger chunks also make citations less precise. To go bigger, swap in
a long-context reranker (such as `BAAI/bge-reranker-v2-m3`) first. The app logs a warning
at startup if `chunk_size` exceeds the embedding model's window. Sizes are estimated
at 4 characters per token rather than with a tokenizer; the estimate only needs to be
roughly right. The cost is speed: bge-m3 is a 568M-parameter model, so ingestion on
CPU is much slower than with MiniLM (a long contract can take minutes). Queries embed
one short question and stay fast.

**Threshold choice.** `min_relevance_score = 0.42` comes from best-chunk cosine scores
on the sample docs with bge-m3 at 512-token chunks:

| Question type | Best-chunk score |
|---|---|
| Answerable, including 10 paraphrases | 0.467 to 0.725 |
| Clearly off-topic (5) | 0.285 to 0.383 |
| On-topic but unanswerable (3) | 0.360, 0.448, 0.551 |

0.42 refuses every off-topic question without an LLM call and keeps every answerable
one, with about 0.04 to 0.05 of margin on each side. On-topic unanswerable questions
("Does Halcyon ship to Canada?" scores 0.55) can't be separated by any threshold, so
the strict prompt (rule 2) handles them. bge-m3 scores run higher than MiniLM's (the
old threshold was 0.25), which is why this must be re-checked whenever
`embedding_model` changes, and ideally on your own documents: raising it trades false
refusals for fewer LLM calls.

**Hybrid retrieval.** `retrieval_mode = "hybrid"` merges BM25 and dense rankings with
reciprocal rank fusion. It helps with exact tokens like part numbers that small
embedding models blur together. RRF only picks and orders the chunks: each returned
chunk keeps its dense cosine score, so the threshold means the same thing in both
modes. It is the default because dense-only retrieval missed too much ground truth
in practice: keyword matches catch exact names, numbers, section references and
defined terms that embeddings blur together. With reranking on, hybrid fills the 15-candidate pool
from both rankings (up to 60 chunks each before fusion), and the cross-encoder then
picks the best 5. The cost is a BM25 index per collection held in memory, rebuilt
when documents are added. Set `retrieval_mode = "dense"` to go back.

**Reranking.** The embedding model scores question and passage separately, so it
can rank a passage that merely shares vocabulary above the one that answers the
question, or push the answer below the top 5. Retrieving 15 candidates and letting a
cross-encoder pick the best 5 widens the net for recall while keeping the prompt
short. The cross-encoder reads question and passage together, which is more
accurate but too slow to run over a whole collection. It adds about 0.2 s per query
on CPU and about 90 MB of model. The threshold still uses dense cosine over all 15
candidates, because cross-encoder scores are unbounded logits with no stable cutoff.
Sources carry both `score` (cosine) and `rerank_score`. On the sample set, chunk-level
recall@5 was already 20/20 without reranking (the corpus is only 8 to 12 chunks), but
reranking moved the answer chunk higher: MRR went from 0.93 to 0.97 with dense
retrieval. Expect the recall gain on larger collections, where the answer can rank
below 5th. Measure it with `run_eval.py`'s context recall on your own documents.

**Embeddings run locally.** bge-m3 runs through ONNX Runtime with no PyTorch and no API
key, so document text never leaves the machine, which matters for confidential legal
material. A hosted legal model such as Voyage's `voyage-law-2` might retrieve better, at
a per-token cost and with documents sent to a third party. The embedder sits behind a
small interface ([app/embeddings.py](app/embeddings.py)), and each collection records
which model built it. Querying with a different `embedding_model` returns 409 instead
of silently giving bad results.

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
  reranker.py        ONNX cross-encoder reranker
  hf.py              Model downloads from the Hugging Face Hub
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
refusal path (asserting the LLM is never called), reranking (candidates in, top-k out), collection isolation (store and
API level), dedup, embedding batching, the embedding-model mismatch check, hybrid
retrieval, and the HTTP endpoints including streaming.
