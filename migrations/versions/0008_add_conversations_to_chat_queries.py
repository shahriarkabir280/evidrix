"""add conversations to chat queries

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-13 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chat_queries", sa.Column("conversation_id", sa.Uuid(), nullable=True))
    op.add_column("chat_queries", sa.Column("search_query", sa.Text(), nullable=True))
    # Chats from before conversations existed each become a conversation of their own.
    op.execute("UPDATE chat_queries SET conversation_id = id")
    op.alter_column("chat_queries", "conversation_id", nullable=False)
    op.create_index(
        "ix_chat_queries_conversation_id_created_at",
        "chat_queries",
        ["conversation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_queries_conversation_id_created_at", table_name="chat_queries")
    op.drop_column("chat_queries", "search_query")
    op.drop_column("chat_queries", "conversation_id")
