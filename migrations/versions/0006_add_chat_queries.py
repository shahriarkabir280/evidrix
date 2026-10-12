"""add chat queries

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-12 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_queries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("document_ids", postgresql.ARRAY(sa.Uuid()), nullable=True),
        sa.Column("retrieval_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("index_version_id", sa.Integer(), nullable=True),
        sa.Column("retrieved", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=True),
        sa.Column("abstain_reason", sa.String(length=32), nullable=True),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("llm_model", sa.String(length=200), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("timings_ms", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            name=op.f("fk_chat_queries_index_version_id_index_versions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_chat_queries_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_queries")),
    )
    op.create_index(
        "ix_chat_queries_user_id_created_at",
        "chat_queries",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_queries_user_id_created_at", table_name="chat_queries")
    op.drop_table("chat_queries")
