"""Dense and keyword retrieval, and Reciprocal Rank Fusion.

Component 2D (dense) · 3A (keyword) · 3B (fusion).
"""

import uuid

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
