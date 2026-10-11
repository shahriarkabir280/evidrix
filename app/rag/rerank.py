"""Reranker protocol and the cross-encoder implementation.

Component 3C.

A bi-encoder (the embedder) turns the question and each chunk into vectors separately, so chunk
vectors can be computed ahead of time and searched in milliseconds. A cross-encoder reads the
question and one chunk together, which judges relevance more accurately but costs one model run
per pair. So it only rescores a small candidate pool that cheaper retrieval already found.
"""

from dataclasses import replace
from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings
from app.db.repositories.chunks import ChunkHit
from app.rag.embeddings import Device, resolve_device


class Reranker(Protocol):
    model_name: str

    def score(self, question: str, passages: list[str]) -> list[float]:
        """One relevance score per passage; higher is more relevant."""
        ...


def rerank(
    reranker: Reranker, question: str, hits: list[ChunkHit], *, top_k: int
) -> list[ChunkHit]:
    """Reorder `hits` by the reranker's scores and keep the best `top_k`.

    Each hit keeps its earlier debug signals plus where it stood before reranking.
    """
    if not hits:
        return []
    scores = reranker.score(question, [passage_for(hit) for hit in hits])
    # Stable sort: equal scores keep their retrieval order.
    ordered = sorted(zip(hits, scores, strict=True), key=lambda pair: -pair[1])[:top_k]
    return [
        replace(
            hit,
            score=float(score),
            rank=rank,
            debug={
                **hit.debug,
                "pre_rerank_rank": hit.rank,
                "pre_rerank_score": hit.score,
                "rerank_score": float(score),
            },
        )
        for rank, (hit, score) in enumerate(ordered, start=1)
    ]


def passage_for(hit: ChunkHit) -> str:
    """What the reranker reads: where the chunk comes from, then its text (as for embedding)."""
    header = hit.section_path or hit.document_title
    return f"{header}\n\n{hit.text}"


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: Device = "auto", batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = resolve_device(device)
        self._model = CrossEncoder(model_name, device=self.device)
        self._batch_size = batch_size

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        # Raw logits: unbounded, only their order matters (and, later, a calibrated threshold).
        scores = self._model.predict(
            [(question, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    return CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)
