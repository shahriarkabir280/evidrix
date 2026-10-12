"""FastAPI dependencies: get_db, get_settings, get_current_user, model and queue providers.

Component 1D · auth added in 5C.
"""

import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_session_factory
from app.rag.embeddings import Embedder, get_embedder
from app.rag.llm import LLMProvider, get_llm
from app.rag.rerank import Reranker, get_reranker
from app.rag.retrieval import RetrievalConfig, load_preset
from app.storage.blob import BlobStore, LocalBlobStore
from app.worker.queue import IngestionQueue, get_ingestion_queue

if TYPE_CHECKING:
    from app.api.schemas import RetrievalOverride

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
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
RerankerDep = Annotated[Reranker, Depends(get_reranker)]
IngestionQueueDep = Annotated[IngestionQueue, Depends(get_ingestion_queue)]


def retrieval_config(settings: Settings, override: "RetrievalOverride | None") -> RetrievalConfig:
    """The server preset with the request's overrides applied, validated as a whole.

    Raises HTTP 422 if the combination is invalid (e.g. top_k above 20 or above candidate_k).
    """
    base = load_preset(settings.retrieval_preset)
    if override is None:
        return base
    try:
        return RetrievalConfig(**{**base.model_dump(), **override.model_dump(exclude_none=True)})
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"code": "invalid_retrieval_config", "message": details},
        ) from exc
