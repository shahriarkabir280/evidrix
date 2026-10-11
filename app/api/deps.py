"""FastAPI dependencies: get_db, get_settings, get_current_user, model providers.

Component 1D · auth added in 5C.
"""

import uuid
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_session_factory
from app.rag.embeddings import Embedder, get_embedder
from app.storage.blob import BlobStore, LocalBlobStore

# Seeded by migration 0001. Replaced by real API-key authentication in component 5C.
DEV_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def get_db() -> Iterator[Session]:
    with get_session_factory()() as session:
        yield session


def get_blob_store(settings: Annotated[Settings, Depends(get_settings)]) -> BlobStore:
    return LocalBlobStore(settings.blob_dir)


def get_current_user_id() -> uuid.UUID:
    return DEV_USER_ID


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[Session, Depends(get_db)]
BlobStoreDep = Annotated[BlobStore, Depends(get_blob_store)]
CurrentUserId = Annotated[uuid.UUID, Depends(get_current_user_id)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
