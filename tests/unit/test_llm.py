import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.rag.llm import (
    LLMBusyError,
    LLMError,
    OpenAICompatibleProvider,
    PromptTooLongError,
    estimate_tokens,
)

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def completion(content: str, *, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3-vl:8b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 14, "total_tokens": 134},
    }


def provider_with(handler, *, context_tokens: int = 8192, **options) -> OpenAICompatibleProvider:
    """A provider whose HTTP calls go to `handler` instead of the network."""
    return OpenAICompatibleProvider(
        base_url="http://ollama.test/v1",
        api_key="ollama",
        model="qwen3-vl:8b-instruct",
        context_tokens=context_tokens,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **options,
    )


def test_sends_an_openai_style_request_with_json_schema():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=completion('{"answer": "ok"}'))

    provider_with(handler).generate("system text", "user text", max_tokens=300, json_schema=SCHEMA)

    path, body = sent[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "qwen3-vl:8b-instruct"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 300
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_result_carries_text_usage_and_finish_reason():
    provider = provider_with(
        lambda r: httpx.Response(200, json=completion("hi", finish_reason="length"))
    )

    result = provider.generate("s", "u", max_tokens=50)

    assert result.text == "hi"
    assert (result.prompt_tokens, result.completion_tokens) == (120, 14)
    assert result.finish_reason == "length"
    assert result.latency_ms >= 0


def test_prompt_that_cannot_fit_is_refused_before_any_request():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion("never"))

    provider = provider_with(handler, context_tokens=1000)

    with pytest.raises(PromptTooLongError):
        provider.generate("system", "x" * 3000, max_tokens=200)
    assert calls == []


def test_server_error_becomes_llm_error():
    provider = provider_with(lambda r: httpx.Response(500, json={"error": "model crashed"}))

    with pytest.raises(LLMError):
        provider.generate("s", "u", max_tokens=10)


def test_unreachable_server_becomes_llm_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError):
        provider_with(handler).generate("s", "u", max_tokens=10)


def test_token_estimate_never_undercounts_measured_text():
    # Measured on real prompts: 3.7-4.2 characters per qwen token.
    text = "word " * 1000
    assert estimate_tokens(text) >= len(text) / 3.7


class SlowServer:
    """Answers after `delay` seconds and records how many requests it handled at once."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self.delay)
        with self._lock:
            self.in_flight -= 1
        return httpx.Response(200, json=completion('{"answer": "ok"}'))


def test_concurrent_calls_wait_for_a_slot_instead_of_piling_onto_the_server():
    server = SlowServer(delay=0.05)
    provider = provider_with(server, max_concurrency=1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: provider.generate("s", "u", max_tokens=10), range(4)))

    assert len(results) == 4
    assert server.max_in_flight == 1


def test_a_call_that_waits_too_long_for_a_slot_is_refused():
    server = SlowServer(delay=0.3)
    provider = provider_with(server, max_concurrency=1, queue_timeout_seconds=0.05)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(provider.generate, "s", "u", max_tokens=10)
        time.sleep(0.05)  # let the first call take the only slot
        second = pool.submit(provider.generate, "s", "u", max_tokens=10)

        assert first.result().text == '{"answer": "ok"}'
        with pytest.raises(LLMBusyError):
            second.result()


def test_a_failed_call_gives_its_slot_back():
    def failing(request):
        return httpx.Response(500, json={"error": "boom"})

    provider = provider_with(failing, max_concurrency=1, queue_timeout_seconds=0.1)

    for _ in range(3):  # would raise LLMBusyError from the second call if the slot leaked
        with pytest.raises(LLMError) as raised:
            provider.generate("s", "u", max_tokens=10)
        assert not isinstance(raised.value, LLMBusyError)
