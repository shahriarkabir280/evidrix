"""Deterministic fakes: FakeEmbedder, FakeLLM, FakeReranker.

Component 2B/2C onward.
"""

import re
import zlib

import numpy as np

_WORD = re.compile(r"[a-z0-9_]+")


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts that share words get similar vectors.

    Good enough to test retrieval plumbing (ranking, filtering, storage) without loading a
    model. It knows nothing about meaning: "car" and "automobile" are unrelated to it.
    """

    model_name = "fake-bag-of-words"
    dim = 64
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
