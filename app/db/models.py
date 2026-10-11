"""SQLAlchemy models for the tables in plan §3.

Component 1B · grows in 2D, 5A, 5C, 6A.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# Fixed by the vector column type. Changing the embedding model to one with another
# dimension needs a migration (and a re-index), not just a setting.
EMBEDDING_DIM = 384

# Postgres text search configuration: lowercases, drops stop words and stems English words
# ("declaring cookies" matches "declare cookie"). See docs/adr/0003.
TEXT_SEARCH_CONFIG = "english"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("owner_id", "content_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(Text)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            native_enum=False,
            create_constraint=True,
            length=20,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        default=DocumentStatus.QUEUED,
    )
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None]
    # Deferred: loaded only when accessed, so listing documents doesn't pull full texts.
    extracted_text: Mapped[str | None] = mapped_column(Text, deferred=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentPage.page_number",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )


class DocumentPage(Base):
    """Maps a PDF page to its character range in Document.extracted_text."""

    __tablename__ = "document_pages"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(primary_key=True)
    char_start: Mapped[int]
    char_end: Mapped[int]

    document: Mapped[Document] = relationship(back_populates="pages")


class IndexVersion(Base):
    """One way of turning documents into searchable chunks: chunking settings + embedding model.

    Several versions can coexist (e.g. 256- vs 400-token chunks for an experiment); every
    query searches exactly one of them.
    """

    __tablename__ = "index_versions"
    __table_args__ = (
        # At most one row may have is_active = true.
        Index(
            "uq_index_versions_one_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_dim: Mapped[int]
    chunking: Mapped[dict[str, Any]] = mapped_column(JSONB)  # ChunkingConfig fields
    contextual_header: Mapped[bool]
    is_active: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "index_version_id", "chunk_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    index_version_id: Mapped[int] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]
    char_start: Mapped[int]
    char_end: Mapped[int]
    page_start: Mapped[int | None]
    page_end: Mapped[int | None]
    section_path: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[np.ndarray] = mapped_column(Vector(EMBEDDING_DIM))
    # Built by Postgres from the section path (weight A) and the text (weight B), so a word in a
    # heading counts more than one in the body. Generated, so it can never go out of sync.
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}', coalesce(section_path, '')), 'A') || "
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}', text), 'B')",
            persisted=True,
        ),
        deferred=True,
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")
