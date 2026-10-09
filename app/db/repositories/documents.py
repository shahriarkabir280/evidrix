"""Document and page queries.

Component 1D.
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Document


def get_by_hash(session: Session, owner_id: uuid.UUID, content_hash: str) -> Document | None:
    return session.scalar(
        select(Document).where(Document.owner_id == owner_id, Document.content_hash == content_hash)
    )


def get_for_owner(session: Session, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document | None:
    return session.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
    )


def list_for_owner(
    session: Session, owner_id: uuid.UUID, *, limit: int, offset: int
) -> tuple[list[Document], int]:
    owned = select(Document).where(Document.owner_id == owner_id)
    total = session.scalar(select(func.count()).select_from(owned.subquery())) or 0
    documents = session.scalars(
        owned.order_by(Document.created_at.desc(), Document.id.desc()).limit(limit).offset(offset)
    ).all()
    return list(documents), total
