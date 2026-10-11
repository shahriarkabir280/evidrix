"""add full-text search column to chunks

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-11 11:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Section path weighs more (A) than body text (B). Adding a stored generated column fills it
# for every existing row as part of this migration.
TSV_EXPRESSION = (
    "setweight(to_tsvector('english', coalesce(section_path, '')), 'A') || "
    "setweight(to_tsvector('english', text), 'B')"
)


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(TSV_EXPRESSION, persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], unique=False, postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_column("chunks", "tsv")
