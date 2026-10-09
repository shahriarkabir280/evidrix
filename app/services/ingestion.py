"""Ingestion pipeline: validate, dedup, extract, chunk, embed, persist.

Component 1D · chunk+embed in 2D · moved to the worker in 5A.
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from typing import BinaryIO

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentPage, DocumentStatus
from app.db.repositories import documents as documents_repo
from app.rag.extract import DocumentKind, ExtractionError, extract
from app.storage.blob import BlobStore, content_key

logger = logging.getLogger(__name__)

READ_CHUNK_BYTES = 1024 * 1024
SNIFF_BYTES = 8192

_KIND_BY_EXTENSION = {
    ".pdf": DocumentKind.PDF,
    ".md": DocumentKind.MARKDOWN,
    ".markdown": DocumentKind.MARKDOWN,
    ".txt": DocumentKind.TEXT,
}
_MIME_TYPE = {
    DocumentKind.PDF: "application/pdf",
    DocumentKind.MARKDOWN: "text/markdown",
    DocumentKind.TEXT: "text/plain",
}


class UploadRejected(Exception):
    """The upload is refused before any document record is created."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class IngestResult:
    document: Document
    created: bool


def read_limited(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read the upload in chunks, stopping as soon as it exceeds max_bytes."""
    chunks: list[bytes] = []
    total = 0
    while chunk := stream.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise UploadRejected(413, "file_too_large", f"The file exceeds {max_bytes} bytes.")
        chunks.append(chunk)
    if total == 0:
        raise UploadRejected(422, "empty_file", "The uploaded file is empty.")
    return b"".join(chunks)


def detect_kind(filename: str, data: bytes) -> DocumentKind:
    """Allow by extension, then confirm by content: a renamed file must not get through."""
    kind = _KIND_BY_EXTENSION.get(PurePath(filename).suffix.lower())
    if kind is None:
        raise UploadRejected(
            415, "unsupported_file_type", "Only PDF, Markdown and plain-text files are supported."
        )
    head = data[:SNIFF_BYTES]
    if kind is DocumentKind.PDF and not head.startswith(b"%PDF-"):
        raise UploadRejected(415, "file_type_mismatch", "The file is named .pdf but is not a PDF.")
    if kind is not DocumentKind.PDF and b"\x00" in head:
        raise UploadRejected(415, "file_type_mismatch", "The file looks binary, not text.")
    return kind


def ingest_upload(
    session: Session,
    blob_store: BlobStore,
    *,
    owner_id: uuid.UUID,
    filename: str,
    data: bytes,
) -> IngestResult:
    filename = _safe_filename(filename)
    kind = detect_kind(filename, data)
    content_hash = content_key(data)

    existing = documents_repo.get_by_hash(session, owner_id, content_hash)
    if existing is not None:
        return IngestResult(existing, created=False)

    # Store the bytes before the row exists, so a document never points at a missing blob.
    blob_store.put(content_hash, data)
    document = Document(
        owner_id=owner_id,
        title=PurePath(filename).stem,
        filename=filename,
        mime_type=_MIME_TYPE[kind],
        size_bytes=len(data),
        content_hash=content_hash,
        storage_key=content_hash,
        status=DocumentStatus.PROCESSING,
    )
    session.add(document)
    try:
        session.commit()
    except IntegrityError:
        # A concurrent upload of the same file committed first; the unique constraint caught it.
        session.rollback()
        winner = documents_repo.get_by_hash(session, owner_id, content_hash)
        if winner is None:
            raise
        return IngestResult(winner, created=False)

    try:
        extracted = extract(data, filename, kind)
        document.title = extracted.title
        document.extracted_text = extracted.text
        document.page_count = len(extracted.pages) if kind is DocumentKind.PDF else None
        document.pages = [
            DocumentPage(
                page_number=span.number, char_start=span.char_start, char_end=span.char_end
            )
            for span in extracted.page_spans()
        ]
        document.status = DocumentStatus.COMPLETED
        session.commit()
    except ExtractionError as exc:
        _mark_failed(session, document, exc.code, str(exc))
    except Exception:
        logger.exception("Unexpected error while ingesting document %s", document.id)
        _mark_failed(session, document, "internal_error", "Unexpected error while processing.")
    return IngestResult(document, created=True)


def _mark_failed(session: Session, document: Document, code: str, message: str) -> None:
    # Discard whatever the failed attempt changed, then record the failure in a fresh
    # transaction, so the document can never be left looking "processing" or "completed".
    session.rollback()
    document.status = DocumentStatus.FAILED
    document.error_code = code
    document.error_message = message[:1000]
    session.commit()


def _safe_filename(filename: str) -> str:
    """Keep only the base name: the client controls this string, and some send full paths."""
    return PurePath(filename.replace("\\", "/")).name[:255] or "upload"
