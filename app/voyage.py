"""Voyage AI embeddings and reranking (Anthropic's recommended embeddings provider).

Anthropic has no embedding model of its own. Voyage runs the models as an API, so
ingestion needs no local model or GPU, but document text is sent to Voyage.

- Embeddings: queries and documents are embedded with different `input_type`s, which
  Voyage recommends for retrieval. Requests are split to stay under the per-request
  limits (1000 texts, and a token budget that depends on the model).
- Reranking: `rerank-2.5` reads up to 32k tokens per query+document, so long chunks
  are scored in full. Its relevance scores are in [0, 1].
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, TypeVar

from app.errors import ProviderError
from app.store import RetrievedChunk

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_TEXTS_PER_REQUEST = 1000
# Total tokens allowed in one embedding request, per model.
MAX_TOKENS_PER_REQUEST: dict[str, int] = {
    "voyage-4-large": 120_000,
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
    "voyage-law-2": 120_000,
}
CONTEXT_TOKENS: dict[str, int] = {"voyage-law-2": 16_000}  # others: 32k
DEFAULT_CONTEXT_TOKENS = 32_000
# Conservative chars-per-token estimate so a request never exceeds the token budget.
CHARS_PER_TOKEN_ESTIMATE = 3
_MISSING_KEY = "VOYAGE_API_KEY is not set; add it to .env (get a key at dash.voyageai.com)"


def _client(api_key: str | None) -> Any:
    import voyageai

    if not api_key:
        raise ProviderError(_MISSING_KEY)
    # The SDK retries rate limits and transient server errors with backoff.
    return voyageai.Client(api_key=api_key, max_retries=3, timeout=120)


def _call(fn: Callable[[], T]) -> T:
    """Run a Voyage call, turning SDK errors into ProviderError with a readable message."""
    import voyageai.error as verr

    try:
        return fn()
    except verr.AuthenticationError as exc:
        raise ProviderError("Voyage rejected the API key; check VOYAGE_API_KEY in .env") from exc
    except verr.RateLimitError as exc:
        raise ProviderError(
            "Voyage rate limit hit; add a payment method to your Voyage account for higher limits"
        ) from exc
    except verr.VoyageError as exc:
        raise ProviderError(f"Voyage error: {exc}") from exc


def token_budget_batches(texts: list[str], max_tokens: int, max_texts: int) -> Iterator[list[str]]:
    """Group texts so each group stays under `max_tokens` (estimated) and `max_texts`."""
    batch: list[str] = []
    tokens = 0
    for text in texts:
        cost = len(text) // CHARS_PER_TOKEN_ESTIMATE + 1
        if batch and (tokens + cost > max_tokens or len(batch) >= max_texts):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += cost
    if batch:
        yield batch


class VoyageEmbedder:
    batch_size = 256  # chunks per progress step; requests are further split by token budget

    def __init__(self, model: str, api_key: str | None, dimensions: int | None = None) -> None:
        self._client = _client(api_key)
        self._model = model
        self._dimensions = dimensions
        # The vector size is part of the identity: collections built at 2048 dims can't be
        # queried with 1024-dim vectors.
        self.model_name = f"{model}@{dimensions}" if dimensions else model
        self.max_input_tokens = CONTEXT_TOKENS.get(model, DEFAULT_CONTEXT_TOKENS)
        # Leave 10% headroom under the request limit for estimation error.
        self._max_request_tokens = int(MAX_TOKENS_PER_REQUEST.get(model, 120_000) * 0.9)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in token_budget_batches(texts, self._max_request_tokens, MAX_TEXTS_PER_REQUEST):
            start = time.perf_counter()
            result = _call(
                lambda: self._client.embed(
                    batch, model=self._model, input_type=input_type, output_dimension=self._dimensions
                )
            )
            logger.info(
                "voyage %s: %d %s(s), %d tokens in %.2fs",
                self._model,
                len(batch),
                input_type,
                result.total_tokens,
                time.perf_counter() - start,
            )
            vectors.extend(result.embeddings)
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class VoyageReranker:
    def __init__(self, model: str, api_key: str | None) -> None:
        self._client = _client(api_key)
        self.model_name = model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        start = time.perf_counter()
        result = _call(
            lambda: self._client.rerank(query, [c.text for c in chunks], model=self.model_name, top_k=top_n)
        )
        logger.info("reranked %d candidates with %s in %.2fs", len(chunks), self.model_name, time.perf_counter() - start)
        # Results come back sorted by relevance, highest first.
        return [replace(chunks[r.index], rerank_score=float(r.relevance_score)) for r in result.results]
