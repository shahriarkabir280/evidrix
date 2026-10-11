"""add index versions and chunks

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-11 09:26:56.290088

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal values, not imports from app code: a migration must keep meaning the same thing
# even after the application's defaults change.
EMBEDDING_DIM = 384
DEFAULT_INDEX_VERSION = {
    "name": "bge-small_c400_o64",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_dim": EMBEDDING_DIM,
    "chunking": {
        "chunk_size_tokens": 400,
        "max_tokens": 480,
        "overlap_tokens": 64,
        "min_chunk_tokens": 100,
        "section_level": 2,
    },
    "contextual_header": True,
    "is_active": True,
}


def upgrade() -> None:
    op.create_table(
        "index_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("chunking", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("contextual_header", sa.Boolean(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_index_versions")),
        sa.UniqueConstraint("name", name=op.f("uq_index_versions_name")),
    )
    op.create_index(
        "uq_index_versions_one_active",
        "index_versions",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("index_version_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            name=op.f("fk_chunks_index_version_id_index_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "document_id",
            "index_version_id",
            "chunk_index",
            name=op.f("uq_chunks_document_id_index_version_id_chunk_index"),
        ),
    )
    # HNSW: approximate nearest-neighbour graph. m = links per node, ef_construction = how
    # hard the build searches for good links. Cosine ops because embeddings are normalized.
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    index_versions = sa.table(
        "index_versions",
        sa.column("name", sa.String()),
        sa.column("embedding_model", sa.String()),
        sa.column("embedding_dim", sa.Integer()),
        sa.column("chunking", postgresql.JSONB()),
        sa.column("contextual_header", sa.Boolean()),
        sa.column("is_active", sa.Boolean()),
    )
    op.bulk_insert(index_versions, [DEFAULT_INDEX_VERSION])


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("uq_index_versions_one_active", table_name="index_versions")
    op.drop_table("index_versions")
