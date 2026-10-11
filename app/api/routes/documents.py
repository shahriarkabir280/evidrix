"""POST/GET/DELETE /documents.

Component 1D · async upload in 5A.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status

from app.api.deps import BlobStoreDep, CurrentUserId, DbSession, EmbedderDep, SettingsDep
from app.api.schemas import DocumentList, DocumentOut, DocumentUploadOut
from app.db.repositories import documents as documents_repo
from app.services.ingestion import UploadRejected, ingest_upload, read_limited

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={200: {"model": DocumentUploadOut, "description": "Already uploaded (duplicate)"}},
)
def upload_document(
    file: UploadFile,
    response: Response,
    session: DbSession,
    blob_store: BlobStoreDep,
    embedder: EmbedderDep,
    owner_id: CurrentUserId,
    settings: SettingsDep,
) -> DocumentUploadOut:
    try:
        data = read_limited(file.file, settings.max_upload_bytes)
        result = ingest_upload(
            session,
            blob_store,
            embedder,
            owner_id=owner_id,
            filename=file.filename or "",
            data=data,
        )
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": str(exc)}) from exc

    if not result.created:
        response.status_code = status.HTTP_200_OK
    document = DocumentOut.model_validate(result.document)
    return DocumentUploadOut(**document.model_dump(), duplicate=not result.created)


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
