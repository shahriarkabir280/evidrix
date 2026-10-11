"""Deterministic fakes: FakeEmbedder, FakeLLM, FakeReranker.

Component 2B/2C onward.
"""

import re
import zlib
from collections.abc import Callable

import numpy as np

from app.rag.llm import LLMResult

_WORD = re.compile(r"[a-z0-9_]+")


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts that share words get similar vectors.

    Good enough to test retrieval plumbing (ranking, filtering, storage) without loading a
    model. It knows nothing about meaning: "car" and "automobile" are unrelated to it.
    """

    model_name = "fake-bag-of-words"
    dim = 384  # matches the vector(384) column, so fake vectors can be stored
    max_tokens = 512

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(text) for text in texts], dtype=np.float32).reshape(
            len(texts), self.dim
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            # crc32, not hash(): Python randomizes str hashes per process (PYTHONHASHSEED),
            # which would make the "same" text embed differently from run to run.
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class FakeLLM:
    """Returns scripted text and records every call, so tests can inspect the prompt sent."""

    model = "fake-llm"

    def __init__(
        self,
        reply: str | list[str] | Callable[[str, str], str] | None = None,
        *,
        finish_reason: str = "stop",
    ) -> None:
        # A list gives one reply per call, in order (e.g. bad JSON first, then a valid reply).
        self._reply = list(reply) if isinstance(reply, list) else reply
        self._finish_reason = finish_reason
        self.calls: list[dict] = []

    def generate(self, system, user, *, max_tokens, json_schema=None) -> LLMResult:
        self.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens, "json_schema": json_schema}
        )
        if callable(self._reply):
            text = self._reply(system, user)
        elif isinstance(self._reply, list):
            text = self._reply.pop(0)
        else:
            text = self._reply or '{"answer": "", "citations": [], "abstain": true}'
        return LLMResult(
            text=text,
            prompt_tokens=len((system + user).split()),
            completion_tokens=len(text.split()),
            model=self.model,
            latency_ms=1.0,
            finish_reason=self._finish_reason,
        )


class FakeReranker:
    """Scores a passage by how many distinct question words it contains."""

    model_name = "fake-word-overlap"

    def score(self, question: str, passages: list[str]) -> list[float]:
        words = set(_WORD.findall(question.lower()))
        return [float(len(words & set(_WORD.findall(p.lower())))) for p in passages]
