"""Index version queries.

Component 2D · create and activate in 8A (chunk-size experiment).
"""

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import IndexVersion


class NoActiveIndexVersionError(Exception):
    pass


def get_active(session: Session) -> IndexVersion:
    index_version = session.scalar(select(IndexVersion).where(IndexVersion.is_active))
    if index_version is None:
        raise NoActiveIndexVersionError("No index version is marked active.")
    return index_version


def get_by_name(session: Session, name: str) -> IndexVersion | None:
    return session.scalar(select(IndexVersion).where(IndexVersion.name == name))


def create_like_active(session: Session, name: str, chunking: dict[str, Any]) -> IndexVersion:
    """A new, inactive version: the active one's embedding model with other chunking settings."""
    active = get_active(session)
    version = IndexVersion(
        name=name,
        embedding_model=active.embedding_model,
        embedding_dim=active.embedding_dim,
        chunking={**active.chunking, **chunking},
        contextual_header=active.contextual_header,
        is_active=False,
    )
    session.add(version)
    session.commit()
    return version


def activate(session: Session, name: str) -> IndexVersion:
    """Make `name` the version every query searches. One transaction: never zero or two active."""
    version = get_by_name(session, name)
    if version is None:
        raise NoActiveIndexVersionError(f"No index version named {name!r}.")
    session.execute(update(IndexVersion).values(is_active=False))
    session.flush()  # the unique partial index allows one active row at a time
    version.is_active = True
    session.commit()
    return version
