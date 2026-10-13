"""Persisted chat queries.

Component 6A.
"""

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ChatQuery


def create(session: Session, owner_id: uuid.UUID, **fields: Any) -> ChatQuery:
    chat = ChatQuery(user_id=owner_id, **fields)
    session.add(chat)
    session.commit()
    return chat


def get_for_owner(session: Session, owner_id: uuid.UUID, chat_id: uuid.UUID) -> ChatQuery | None:
    return session.scalar(
        select(ChatQuery).where(ChatQuery.id == chat_id, ChatQuery.user_id == owner_id)
    )


def conversation_turns(
    session: Session, owner_id: uuid.UUID, conversation_id: uuid.UUID, limit: int
) -> list[ChatQuery]:
    """The last `limit` chats of a conversation that didn't fail, oldest first."""
    newest = session.scalars(
        select(ChatQuery)
        .where(
            ChatQuery.user_id == owner_id,
            ChatQuery.conversation_id == conversation_id,
            ChatQuery.error_code.is_(None),
        )
        .order_by(ChatQuery.created_at.desc())
        .limit(limit)
    ).all()
    return list(reversed(newest))


def list_for_owner(
    session: Session,
    owner_id: uuid.UUID,
    *,
    limit: int,
    offset: int,
    conversation_id: uuid.UUID | None = None,
) -> tuple[list[ChatQuery], int]:
    owned = select(ChatQuery).where(ChatQuery.user_id == owner_id)
    if conversation_id is not None:
        owned = owned.where(ChatQuery.conversation_id == conversation_id)
    total = session.scalar(select(func.count()).select_from(owned.subquery())) or 0
    chats = session.scalars(
        owned.order_by(ChatQuery.created_at.desc(), ChatQuery.id.desc()).limit(limit).offset(offset)
    ).all()
    return list(chats), total
