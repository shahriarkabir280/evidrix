"""LLMProvider protocol and the OpenAI-compatible (Ollama) implementation.

Component 2E · Plan: Part 2, step 8.

Only this module imports the vendor SDK; the rest of the app depends on LLMProvider.
"""

import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from app.core.config import get_settings

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# Measured with qwen3 on our prompts: 3.7-4.2 characters per token (code-heavy text is lower).
# Dividing by 3.0 overestimates the token count, so the guard errs on the safe side.
CHARS_PER_TOKEN_ESTIMATE = 3.0
# Chat templates wrap each message in role markers; a small allowance covers them.
MESSAGE_OVERHEAD_TOKENS = 16


class LLMError(Exception):
    """The model could not be reached, timed out, or returned an error."""


class PromptTooLongError(Exception):
    """The prompt would not fit in the model's context window, so it was not sent."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_ms: float
    finish_reason: str  # "stop", or "length" when the output hit max_tokens


class LLMProvider(Protocol):
    model: str

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        context_tokens: int,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        temperature: float = 0.0,
        http_client: "httpx.Client | None" = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.context_tokens = context_tokens
        self._temperature = temperature
        # The SDK already retries connection errors, 429 and 5xx with backoff; we don't add
        # a second retry layer on top, which would multiply the worst-case wait.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client,
        )

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        import openai

        # Ollama silently drops the start of a prompt that overflows its context window.
        # Refuse up front instead, so truncation is an error rather than a quiet wrong answer.
        estimated = estimate_tokens(system) + estimate_tokens(user) + 2 * MESSAGE_OVERHEAD_TOKENS
        if estimated + max_tokens > self.context_tokens:
            raise PromptTooLongError(
                f"Prompt is about {estimated} tokens; with {max_tokens} output tokens it exceeds "
                f"the {self.context_tokens}-token context window."
            )

        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
                max_tokens=max_tokens,
                **extra,
            )
        except openai.APIError as exc:
            raise LLMError(f"LLM request to {self.model} failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choice = response.choices[0]
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        if prompt_tokens + max_tokens > self.context_tokens:
            logger.warning(
                "Prompt used %d of %d context tokens; the estimate let it through",
                prompt_tokens,
                self.context_tokens,
            )
        return LLMResult(
            text=choice.message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=response.model or self.model,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "stop",
        )


@lru_cache
def get_llm() -> LLMProvider:
    settings = get_settings()
    return OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        context_tokens=settings.llm_context_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
