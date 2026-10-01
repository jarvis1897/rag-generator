"""Thin LLM client wrapper. The provider is chosen by `LLM_PROVIDER`.

Only Anthropic is implemented; another provider only needs a class with the
same two methods and a branch in `create_llm`.
"""

import logging
from collections.abc import Iterator
from typing import Literal, Protocol

from app.config import Settings

logger = logging.getLogger(__name__)

Effort = Literal["low", "medium", "high"]


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(
        self, system: str, messages: list[dict[str, str]], max_tokens: int, effort: Effort | None = None
    ) -> str: ...

    def stream(
        self, system: str, messages: list[dict[str, str]], max_tokens: int, effort: Effort | None = None
    ) -> Iterator[str]: ...


class AnthropicLLM:
    # "default" lets the API pick a fallback model by refusal category.
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, refusal_fallback: bool) -> None:
        import anthropic  # reads ANTHROPIC_API_KEY from the environment; never logged

        self._client = anthropic.Anthropic()
        self._model = model
        self._refusal_fallback = refusal_fallback

    def _params(self, system: str, messages: list[dict[str, str]], max_tokens: int, effort: Effort | None):
        params: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if effort:
            params["output_config"] = {"effort": effort}
        if self._refusal_fallback:
            params["betas"] = [self._FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def stream(
        self, system: str, messages: list[dict[str, str]], max_tokens: int, effort: Effort | None = None
    ) -> Iterator[str]:
        import anthropic

        try:
            with self._client.beta.messages.stream(**self._params(system, messages, max_tokens, effort)) as s:
                yield from s.text_stream
                final = s.get_final_message()
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the LLM provider: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("LLM provider rate limit hit, try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM provider error {exc.status_code}: {exc.message}") from exc

        if final.stop_reason == "refusal":
            raise LLMError("the model declined to answer this request")
        if final.stop_reason == "max_tokens":
            logger.warning("LLM output hit max_tokens (%d) and was truncated", max_tokens)

    def complete(
        self, system: str, messages: list[dict[str, str]], max_tokens: int, effort: Effort | None = None
    ) -> str:
        return "".join(self.stream(system, messages, max_tokens, effort))


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.llm_refusal_fallback)
    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
