"""Index version queries.

Component 2D.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import IndexVersion


class NoActiveIndexVersionError(Exception):
    pass


def get_active(session: Session) -> IndexVersion:
    index_version = session.scalar(select(IndexVersion).where(IndexVersion.is_active))
    if index_version is None:
        raise NoActiveIndexVersionError("No index version is marked active.")
    return index_version
