"""Chunk storage plus dense and keyword retrieval queries.

Component 2D · keyword search in 3A.
"""

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus

# Candidates HNSW examines per query. Higher = closer to exact search, slower.
# The pgvector default is 40; it must be at least the number of results requested.
HNSW_EF_SEARCH = 100


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    text: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    score: float  # cosine similarity, higher is better
    rank: int  # 1-based


def search_dense(
    session: Session,
    *,
    owner_id: uuid.UUID,
    index_version_id: int,
    query_vector: np.ndarray,
    k: int,
    document_ids: list[uuid.UUID] | None = None,
    exact: bool = False,
) -> list[ChunkHit]:
    """Nearest chunks by cosine distance, restricted to the owner's completed documents.

    The filters live in SQL, so another user's chunks are never even candidates.
    """
    if document_ids is not None and not document_ids:
        return []

    distance = Chunk.embedding.cosine_distance(query_vector)
    query = (
        select(
            Chunk.id,
            Chunk.document_id,
            Document.title,
            Chunk.chunk_index,
            Chunk.text,
            Chunk.section_path,
            Chunk.page_start,
            Chunk.page_end,
            distance.label("distance"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Document.owner_id == owner_id,
            Document.status == DocumentStatus.COMPLETED,
            Chunk.index_version_id == index_version_id,
        )
        .order_by(distance)
        .limit(k)
    )
    if document_ids is not None:
        query = query.where(Chunk.document_id.in_(document_ids))

    if exact:
        # Skip the HNSW index: a full scan returns the true nearest neighbours.
        session.execute(text("SET LOCAL enable_indexscan = off"))
    else:
        # With a WHERE clause, HNSW may find fewer than k matching rows among the candidates
        # it examines. Iterative scan keeps searching until it has enough (pgvector >= 0.8).
        session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        session.execute(text(f"SET LOCAL hnsw.ef_search = {max(HNSW_EF_SEARCH, k)}"))

    rows = session.execute(query).all()
    # relaxed_order may return rows slightly out of order; sort by the real distance.
    rows = sorted(rows, key=lambda row: row.distance)
    return [
        ChunkHit(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.title,
            chunk_index=row.chunk_index,
            text=row.text,
            section_path=row.section_path,
            page_start=row.page_start,
            page_end=row.page_end,
            score=1.0 - float(row.distance),
            rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]
