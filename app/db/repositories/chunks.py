"""Chunk storage plus dense and keyword retrieval queries.

Component 2D · keyword search in 3A.
"""

import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import Select, Text, cast, func, literal_column, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.orm import Session

from app.db.models import TEXT_SEARCH_CONFIG, Chunk, Document, DocumentStatus

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
    score: float  # higher is better: cosine (dense), ts_rank_cd (keyword) or RRF (hybrid)
    rank: int  # 1-based
    # How each retrieval stage saw this chunk, e.g. {"dense_rank": 2, "keyword_score": 0.1}.
    debug: dict[str, float] = field(default_factory=dict)
    # Position of the chunk in Document.extracted_text (used to match evaluation labels).
    char_start: int | None = None
    char_end: int | None = None


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
    """Nearest chunks by cosine distance, restricted to the owner's completed documents."""
    if document_ids is not None and not document_ids:
        return []

    distance = Chunk.embedding.cosine_distance(query_vector)
    query = _searchable(owner_id, index_version_id, document_ids, distance.label("distance"))
    query = query.order_by(distance).limit(k)

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
        _hit(row, score=1.0 - float(row.distance), rank=rank) for rank, row in enumerate(rows, 1)
    ]


def search_keyword(
    session: Session,
    *,
    owner_id: uuid.UUID,
    index_version_id: int,
    query_text: str,
    k: int,
    document_ids: list[uuid.UUID] | None = None,
) -> list[ChunkHit]:
    """Chunks containing any of the question's words, best ts_rank_cd first.

    The words are OR-ed: requiring all of them (what websearch_to_tsquery does) makes a natural
    question with one extra word match nothing.
    """
    if document_ids is not None and not document_ids:
        return []

    tsquery = or_tsquery(query_text)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery)
    query = _searchable(owner_id, index_version_id, document_ids, rank.label("rank_score"))
    query = query.where(Chunk.tsv.op("@@")(tsquery)).order_by(rank.desc(), Chunk.id).limit(k)
    rows = session.execute(query).all()
    return [_hit(row, score=float(row.rank_score), rank=rank) for rank, row in enumerate(rows, 1)]


def or_tsquery(query_text: str):
    """A tsquery matching ANY word of `query_text`, built safely from untrusted input.

    plainto_tsquery accepts arbitrary text (no syntax errors from "&", "|", "!" or quotes),
    drops stop words, stems, and joins the lexemes with "&". Swapping "&" for "|" makes it OR.
    """
    config = literal_column(f"'{TEXT_SEARCH_CONFIG}'::regconfig")
    and_query = cast(func.plainto_tsquery(config, query_text), Text)
    return cast(func.replace(and_query, " & ", " | "), TSQUERY)


def _searchable(
    owner_id: uuid.UUID, index_version_id: int, document_ids: list[uuid.UUID] | None, score
) -> Select:
    """The columns of a hit plus the filters every search must apply, in one place.

    The filters live in SQL, so another user's chunks are never even candidates.
    """
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
            Chunk.char_start,
            Chunk.char_end,
            score,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Document.owner_id == owner_id,
            Document.status == DocumentStatus.COMPLETED,
            Chunk.index_version_id == index_version_id,
        )
    )
    if document_ids is not None:
        query = query.where(Chunk.document_id.in_(document_ids))
    return query


def _hit(row, *, score: float, rank: int) -> ChunkHit:
    return ChunkHit(
        chunk_id=row.id,
        document_id=row.document_id,
        document_title=row.title,
        chunk_index=row.chunk_index,
        text=row.text,
        section_path=row.section_path,
        page_start=row.page_start,
        page_end=row.page_end,
        score=score,
        rank=rank,
        char_start=row.char_start,
        char_end=row.char_end,
    )
