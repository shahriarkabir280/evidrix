"""Embedder protocol and the sentence-transformers implementation.

Component 2C · Plan: Part 2, step 4.
"""

from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from app.core.config import get_settings

Device = Literal["auto", "cpu", "mps", "cuda"]


class Embedder(Protocol):
    model_name: str
    dim: int
    max_tokens: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Shape (len(texts), dim), float32, L2-normalized."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Shape (dim,), float32, L2-normalized."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens as this model's tokenizer counts them, without special tokens."""
        ...


def resolve_device(preference: Device) -> str:
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        query_instruction: str = "",
        device: Device = "auto",
        batch_size: int = 32,
    ) -> None:
        # Imported here so modules that only need the Embedder protocol don't load torch.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_device(device)
        self._model = SentenceTransformer(model_name, device=self.device)
        self._query_instruction = query_instruction
        self._batch_size = batch_size
        self.dim: int = self._model.get_embedding_dimension()
        self.max_tokens: int = self._model.max_seq_length

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_instruction + text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def _encode(self, texts: list[str]) -> np.ndarray:
        # Normalized vectors make the dot product equal to cosine similarity.
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)


@lru_cache
def get_embedder() -> Embedder:
    """The process-wide embedder. Loading the model takes seconds, so it happens once."""
    settings = get_settings()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
