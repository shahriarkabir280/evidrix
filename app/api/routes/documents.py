"""POST/GET/DELETE /documents.

Component 1D · async upload in 5A.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, UploadFile, status

from app.api.deps import BlobStoreDep, CurrentUserId, DbSession, IngestionQueueDep, SettingsDep
from app.api.rate_limit import rate_limit
from app.api.schemas import DocumentList, DocumentOut, DocumentUploadOut
from app.core.logging import current_request_id
from app.db.repositories import documents as documents_repo
from app.db.repositories import jobs as jobs_repo
from app.services.ingestion import UploadRejected, accept_upload, mark_failed, read_limited
from app.worker.queue import QueueUnavailableError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit("upload"))],
    responses={200: {"model": DocumentUploadOut, "description": "Already uploaded (duplicate)"}},
)
def upload_document(
    file: UploadFile,
    response: Response,
    session: DbSession,
    blob_store: BlobStoreDep,
    queue: IngestionQueueDep,
    owner_id: CurrentUserId,
    settings: SettingsDep,
) -> DocumentUploadOut:
    """Accept the file and queue it for processing; returns before any extraction happens."""
    try:
        data = read_limited(file.file, settings.max_upload_bytes)
        result = accept_upload(
            session,
            blob_store,
            owner_id=owner_id,
            filename=file.filename or "",
            data=data,
            max_attempts=settings.ingestion_max_attempts,
            request_id=current_request_id(),
        )
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": str(exc)}) from exc

    if result.job is None:
        # A duplicate with nothing to do: 200, and the job that processed it (if any).
        response.status_code = status.HTTP_200_OK
        latest = jobs_repo.latest_for_document(session, owner_id, result.document.id)
        job_id = latest.id if latest else None
    else:
        # Enqueue only after the commit in accept_upload: the worker must find the job's row.
        try:
            queue.enqueue(result.job.id)
        except QueueUnavailableError as exc:
            mark_failed(session, result.job, "queue_unavailable", "The job queue is unavailable.")
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                {"code": "queue_unavailable", "message": "Try the upload again later."},
            ) from exc
        job_id = result.job.id
    document = DocumentOut.model_validate(result.document)
    return DocumentUploadOut(**document.model_dump(), job_id=job_id, duplicate=result.duplicate)


@router.get("")
def list_documents(
    session: DbSession,
    owner_id: CurrentUserId,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    documents, total = documents_repo.list_for_owner(session, owner_id, limit=limit, offset=offset)
    return DocumentList(
        items=[DocumentOut.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}")
def get_document(
    document_id: uuid.UUID, session: DbSession, owner_id: CurrentUserId
) -> DocumentOut:
    document = documents_repo.get_for_owner(session, owner_id, document_id)
    if document is None:
        # Same answer for "doesn't exist" and "belongs to someone else": don't leak existence.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            {"code": "document_not_found", "message": "Document not found."},
        )
    return DocumentOut.model_validate(document)
