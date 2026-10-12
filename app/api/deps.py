"""FastAPI dependencies: get_db, get_settings, get_current_user, model and queue providers.

Component 1D · auth added in 5C.
"""

import uuid
from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import looks_like_api_key
from app.db.repositories import users as users_repo
from app.db.session import get_session_factory
from app.rag.embeddings import Embedder, get_embedder
from app.rag.llm import LLMProvider, get_llm
from app.rag.rerank import Reranker, get_reranker
from app.rag.retrieval import RetrievalConfig, load_preset
from app.storage.blob import BlobStore, LocalBlobStore
from app.worker.queue import IngestionQueue, get_ingestion_queue

if TYPE_CHECKING:
    from app.api.schemas import RetrievalOverride

# auto_error=False: a missing header reaches get_current_user_id, which answers with our own
# 401 body (FastAPI's default would be a 403 with a different shape).
_bearer = HTTPBearer(auto_error=False, description="An API key: `Authorization: Bearer evx_…`")


def get_db() -> Iterator[Session]:
    with get_session_factory()() as session:
        yield session


@lru_cache
def get_redis() -> Redis:
    # Short timeouts: readiness probes and the rate limiter must not hang on a slow Redis.
    return Redis.from_url(get_settings().redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)


def get_blob_store(settings: Annotated[Settings, Depends(get_settings)]) -> BlobStore:
    return LocalBlobStore(settings.blob_dir)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[Session, Depends(get_db)]


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: DbSession,
    request: Request,
) -> uuid.UUID:
    """The user the request's API key belongs to; 401 for a missing, unknown or revoked key.

    The three cases get the same answer, so a caller can't learn whether a key ever existed.
    """
    key = credentials.credentials if credentials else ""
    user_id = users_repo.user_id_for_key(session, key) if looks_like_api_key(key) else None
    if user_id is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            {"code": "invalid_api_key", "message": "Missing, invalid or revoked API key."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.user_id = user_id  # for the access log line (app/api/middleware.py)
    return user_id


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
