"""add display title to documents

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-18 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("display_title", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "display_title")
