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


_NO_CREDENTIALS = "no Anthropic credentials configured; set ANTHROPIC_API_KEY in the environment or .env"


class AnthropicLLM:
    # "default" lets the API pick a fallback model by refusal category.
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, refusal_fallback: bool, api_key: str | None = None) -> None:
        import anthropic

        # Pass the key explicitly: pydantic-settings reads .env but does not export it
        # to os.environ, where the SDK would otherwise look. The key is never logged.
        try:
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        except anthropic.AnthropicError as exc:  # e.g. a configured credentials profile that is missing
            raise LLMError(f"{_NO_CREDENTIALS}: {exc}") from exc
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
        except anthropic.AuthenticationError as exc:
            raise LLMError("LLM provider rejected the API key; check ANTHROPIC_API_KEY") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM provider error {exc.status_code}: {exc.message}") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"LLM provider error: {exc}") from exc
        except TypeError as exc:
            # The SDK raises a bare TypeError when it finds no credentials at all.
            if "authentication" not in str(exc).lower():
                raise
            raise LLMError(_NO_CREDENTIALS) from exc

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
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicLLM(settings.llm_model, settings.llm_refusal_fallback, api_key=key)
    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
