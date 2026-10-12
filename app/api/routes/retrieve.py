"""POST /retrieve: retrieval only, with per-stage debug scores.

Component 3D.

Lets you inspect what the model would see for a question, and why each chunk ranked where it did,
without spending an LLM call.
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import (
    CurrentUserId,
    DbSession,
    EmbedderDep,
    RerankerDep,
    SettingsDep,
    retrieval_config,
)
from app.api.rate_limit import rate_limit
from app.api.routes.chat import error, retrieved_out
from app.api.schemas import RetrievalConfigOut, RetrieveRequest, RetrieveResponse
from app.db.repositories import index_versions as index_versions_repo
from app.rag.retrieval import retrieve
from app.services.answering import DocumentNotFoundError, DocumentNotReadyError, check_documents
from app.services.ingestion import ensure_compatible

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", dependencies=[Depends(rate_limit("retrieve"))])
def retrieve_chunks(
    request: RetrieveRequest,
    session: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    owner_id: CurrentUserId,
    settings: SettingsDep,
) -> RetrieveResponse:
    config = retrieval_config(settings, request.retrieval)
    document_ids = request.document_ids
    if document_ids is not None:
        document_ids = list(dict.fromkeys(document_ids))
        try:
            check_documents(session, owner_id, document_ids)
        except DocumentNotFoundError as exc:
            raise error(status.HTTP_404_NOT_FOUND, "document_not_found", exc) from exc
        except DocumentNotReadyError as exc:
            raise error(status.HTTP_409_CONFLICT, "document_not_ready", exc) from exc

    index_version = index_versions_repo.get_active(session)
    ensure_compatible(index_version, embedder)
    result = retrieve(
        session,
        embedder,
        reranker,
        config,
        owner_id=owner_id,
        index_version_id=index_version.id,
        question=request.question,
        document_ids=document_ids,
    )
    return RetrieveResponse(
        hits=[retrieved_out(hit) for hit in result.hits],
        timings_ms=result.timings_ms,
        config=RetrievalConfigOut(
            retrieval=config,
            index_version=index_version.name,
            embedding_model=embedder.model_name,
            reranker_model=reranker.model_name if config.rerank else None,
        ),
    )
