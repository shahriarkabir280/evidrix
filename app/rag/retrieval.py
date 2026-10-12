"""Dense and keyword retrieval, and Reciprocal Rank Fusion.

Component 2D (dense) · 3A (keyword) · 3B (fusion) · 3D (configurable pipeline).
"""

import tomllib
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.core.timing import StageTimer
from app.db.repositories import chunks as chunks_repo
from app.db.repositories.chunks import ChunkHit
from app.rag.embeddings import Embedder
from app.rag.rerank import Reranker, rerank

PRESETS_DIR = Path(__file__).resolve().parents[2] / "configs" / "retrieval"


def dense_search(
    session: Session,
    embedder: Embedder,
    *,
    owner_id: uuid.UUID,
    index_version_id: int,
    question: str,
    k: int = 5,
    document_ids: list[uuid.UUID] | None = None,
) -> list[ChunkHit]:
    """Embed the question and return the k most similar chunks the owner may see."""
    return chunks_repo.search_dense(
        session,
        owner_id=owner_id,
        index_version_id=index_version_id,
        query_vector=embedder.embed_query(question),
        k=k,
        document_ids=document_ids,
    )


def keyword_search(
    session: Session,
    *,
    owner_id: uuid.UUID,
    index_version_id: int,
    question: str,
    k: int = 5,
    document_ids: list[uuid.UUID] | None = None,
) -> list[ChunkHit]:
    """Chunks sharing words with the question (Postgres full-text search)."""
    return chunks_repo.search_keyword(
        session,
        owner_id=owner_id,
        index_version_id=index_version_id,
        query_text=question,
        k=k,
        document_ids=document_ids,
    )


# The constant from the original RRF paper (Cormack et al., 2009). Larger values flatten the
# curve, so agreement between lists matters more than being first in one of them.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: dict[str, list[ChunkHit]], *, limit: int, rrf_k: int = RRF_K
) -> list[ChunkHit]:
    """Merge ranked lists using positions only: score = sum of 1 / (rrf_k + rank).

    Scores from different retrievers (cosine ~0.8, ts_rank_cd ~0.1) are on unrelated scales, so
    they are never added together; ranks are comparable across lists.
    """
    fused: dict[uuid.UUID, float] = defaultdict(float)
    first_seen: dict[uuid.UUID, ChunkHit] = {}
    debug: dict[uuid.UUID, dict[str, float]] = defaultdict(dict)
    for name, hits in rankings.items():
        for hit in hits:
            fused[hit.chunk_id] += 1.0 / (rrf_k + hit.rank)
            first_seen.setdefault(hit.chunk_id, hit)
            debug[hit.chunk_id][f"{name}_rank"] = hit.rank
            debug[hit.chunk_id][f"{name}_score"] = hit.score

    # Ties (same score) are broken by chunk id, so the order is the same on every run.
    order = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], str(chunk_id)))[:limit]
    return [
        replace(
            first_seen[chunk_id],
            score=fused[chunk_id],
            rank=rank,
            debug={**debug[chunk_id], "rrf_score": fused[chunk_id]},
        )
        for rank, chunk_id in enumerate(order, start=1)
    ]


def hybrid_search(
    session: Session,
    embedder: Embedder,
    *,
    owner_id: uuid.UUID,
    index_version_id: int,
    question: str,
    k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = RRF_K,
    document_ids: list[uuid.UUID] | None = None,
) -> list[ChunkHit]:
    """Dense and keyword search each fetch `candidate_k` chunks; RRF keeps the best `k`."""
    common = {
        "owner_id": owner_id,
        "index_version_id": index_version_id,
        "question": question,
        "k": candidate_k,
        "document_ids": document_ids,
    }
    return reciprocal_rank_fusion(
        {
            "dense": dense_search(session, embedder, **common),
            "keyword": keyword_search(session, **common),
        },
        limit=k,
        rrf_k=rrf_k,
    )


class RetrievalConfig(BaseModel):
    """How to find evidence for a question. Presets live in configs/retrieval/*.toml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["dense", "keyword", "hybrid"] = "hybrid"
    rerank: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    # How many chunks each retriever fetches before fusion and/or reranking.
    candidate_k: int = Field(default=20, ge=1, le=100)
    rrf_k: int = Field(default=RRF_K, ge=1, le=1000)
    # Abstain without calling the model when the best chunk scores below this. It is on the scale
    # of the pipeline's final score (cosine for dense, RRF for hybrid, reranker logit with
    # rerank), so each preset needs its own value, calibrated on the dev split (evals/
    # calibrate_abstention.py). None disables the gate.
    min_score: float | None = None
    # Leave out chunks in a "References" / "Bibliography" section (research papers): citation
    # lists are keyword-dense but rarely contain an answer.
    exclude_references: bool = False

    @model_validator(mode="after")
    def _pool_holds_the_results(self) -> "RetrievalConfig":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self


class UnknownPresetError(ValueError):
    pass


def load_preset(name: str) -> RetrievalConfig:
    path = PRESETS_DIR / f"{name}.toml"
    # Only plain names: a preset is chosen by name, never by a path.
    if not name.replace("_", "").isalnum() or not path.is_file():
        available = sorted(p.stem for p in PRESETS_DIR.glob("*.toml"))
        raise UnknownPresetError(f"Unknown retrieval preset {name!r}; available: {available}")
    return RetrievalConfig(**tomllib.loads(path.read_text()))


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[ChunkHit]
    timings_ms: dict[str, float]


def retrieve(
    session: Session,
    embedder: Embedder,
    reranker: Reranker,
    config: RetrievalConfig,
    *,
    owner_id: uuid.UUID,
    index_version_id: int,
    question: str,
    document_ids: list[uuid.UUID] | None = None,
    timer: StageTimer | None = None,
) -> RetrievalResult:
    """Run the pipeline `config` describes and time each stage (into `timer`, if given).

    With reranking, each retriever fetches the larger candidate pool and the reranker picks the
    final top_k; without it, retrieval returns top_k directly.
    """
    timer = timer or StageTimer()
    fetch = config.candidate_k if (config.rerank or config.mode == "hybrid") else config.top_k
    common = {
        "owner_id": owner_id,
        "index_version_id": index_version_id,
        "k": fetch,
        "document_ids": document_ids,
        "exclude_references": config.exclude_references,
    }
    legs: dict[str, list[ChunkHit]] = {}

    if config.mode in ("dense", "hybrid"):
        with timer.stage("embed"):
            query_vector = embedder.embed_query(question)
        with timer.stage("retrieve_dense"):
            legs["dense"] = chunks_repo.search_dense(session, query_vector=query_vector, **common)
    if config.mode in ("keyword", "hybrid"):
        with timer.stage("retrieve_keyword"):
            legs["keyword"] = chunks_repo.search_keyword(session, query_text=question, **common)

    if config.mode == "hybrid":
        with timer.stage("fuse"):
            pool_size = config.candidate_k if config.rerank else config.top_k
            hits = reciprocal_rank_fusion(legs, limit=pool_size, rrf_k=config.rrf_k)
    else:
        (hits,) = legs.values()

    if config.rerank:
        with timer.stage("rerank"):
            hits = rerank(reranker, question, hits, top_k=config.top_k)
    return RetrievalResult(hits=hits[: config.top_k], timings_ms=dict(timer.timings_ms))
