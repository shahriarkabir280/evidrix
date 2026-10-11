"""Dense and keyword retrieval, and Reciprocal Rank Fusion.

Component 2D (dense) · 3A (keyword) · 3B (fusion).
"""

import uuid
from collections import defaultdict
from dataclasses import replace

from sqlalchemy.orm import Session

from app.db.repositories import chunks as chunks_repo
from app.db.repositories.chunks import ChunkHit
from app.rag.embeddings import Embedder


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
