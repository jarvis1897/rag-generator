"""Errors shared across modules."""


class ProviderError(RuntimeError):
    """A hosted model provider (embeddings, reranking) failed or is misconfigured.

    Messages are safe to show to users: they never contain API keys.
    """
