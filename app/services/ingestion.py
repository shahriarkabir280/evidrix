"""Ingestion pipeline: validate, dedup, extract, chunk, embed, persist.

Component 1D · chunk+embed in 2D · split into accept (API) and process (worker) in 5A.

The API only accepts an upload: it validates, stores the bytes and records a document and a job
in one transaction. The slow part (extract, chunk, embed) runs later in the worker, in
run_ingestion_job. ingest_upload does both in one call, for the CLI and tests.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePath
from typing import BinaryIO

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.db.models import (
    Chunk,
    Document,
    DocumentPage,
    DocumentStatus,
    IndexVersion,
    IngestionJob,
    JobStatus,
)
from app.db.repositories import documents as documents_repo
from app.db.repositories import index_versions as index_versions_repo
from app.db.repositories import jobs as jobs_repo
from app.rag.chunking import ChunkingConfig, chunk_document
from app.rag.embeddings import Embedder, text_for_embedding
from app.rag.extract import (
    PAGE_SEPARATOR,
    DocumentKind,
    ExtractedDocument,
    ExtractionError,
    PageText,
    extract,
)
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
_KIND_BY_MIME_TYPE = {mime: kind for kind, mime in _MIME_TYPE.items()}
DEFAULT_MAX_ATTEMPTS = 4

# Errors that may go away if the same job runs again a little later: a slow or unreachable
# embedding service, a database restart. Anything else is treated as permanent, because
# retrying a bug or a broken file only repeats the same failure.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, OperationalError)


class IndexConfigurationError(Exception):
    """The running embedder doesn't match the active index version (a deployment mistake)."""


class UploadRejected(Exception):
    """The upload is refused before any document record is created."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class UploadResult:
    document: Document
    job: IngestionJob | None  # the job to enqueue; None when there is nothing to process
    duplicate: bool = False  # the file was already uploaded (the document is the earlier one)

    @property
    def created(self) -> bool:
        return not self.duplicate


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


def accept_upload(
    session: Session,
    blob_store: BlobStore,
    *,
    owner_id: uuid.UUID,
    filename: str,
    data: bytes,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    request_id: str | None = None,
) -> UploadResult:
    """Validate and record the upload; commit a queued document and its job together.

    The caller enqueues the job only after this returns, so the worker can never receive a job
    id whose row isn't committed yet. Uploading a file again returns the earlier document; if
    that one failed, it gets a new job, so uploading again is how a failed document is retried.
    """
    filename = _safe_filename(filename)
    kind = detect_kind(filename, data)
    content_hash = content_key(data)
    index_version = index_versions_repo.get_active(session)

    existing = documents_repo.get_by_hash(session, owner_id, content_hash)
    if existing is not None:
        if existing.status is not DocumentStatus.FAILED:
            return UploadResult(existing, job=None, duplicate=True)
        existing.status = DocumentStatus.QUEUED
        existing.error_code = existing.error_message = None
        retry = IngestionJob(
            document=existing,
            index_version_id=index_version.id,
            status=JobStatus.QUEUED,
            max_attempts=max_attempts,
            request_id=request_id,
        )
        session.add(retry)
        session.commit()
        return UploadResult(existing, retry, duplicate=True)

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
        status=DocumentStatus.QUEUED,
    )
    job = IngestionJob(
        document=document,
        index_version_id=index_version.id,
        status=JobStatus.QUEUED,
        max_attempts=max_attempts,
        request_id=request_id,
    )
    session.add_all([document, job])
    try:
        session.commit()
    except IntegrityError:
        # A concurrent upload of the same file committed first; the unique constraint caught it.
        session.rollback()
        winner = documents_repo.get_by_hash(session, owner_id, content_hash)
        if winner is None:
            raise
        return UploadResult(winner, job=None, duplicate=True)
    return UploadResult(document, job)


def run_ingestion_job(
    session: Session, blob_store: BlobStore, embedder: Embedder, job_id: uuid.UUID
) -> JobStatus | None:
    """Process one job: extract, chunk and embed its document. Returns the job's new status.

    Safe to call more than once for the same job: only a claimable job is processed, and all of
    its results are written in a single transaction, so there is never a half-indexed document.
    RETRYING means a transient error with attempts left: the caller should run the job again
    later (the RQ worker does this with backoff).
    """
    job = jobs_repo.claim(session, job_id)
    if job is None:
        logger.info("Job %s is not claimable (duplicate delivery or already done)", job_id)
        return None
    document = job.document
    document.status = DocumentStatus.PROCESSING
    session.commit()

    try:
        index_version = session.get_one(IndexVersion, job.index_version_id)
        ensure_compatible(index_version, embedder)
        kind = _KIND_BY_MIME_TYPE[document.mime_type]
        extracted = extract(blob_store.get(document.storage_key), document.filename, kind)

        # Clear anything an earlier attempt left, then write the new results.
        document.pages.clear()
        document.chunks = [c for c in document.chunks if c.index_version_id != index_version.id]
        session.flush()
        document.title = extracted.title
        document.extracted_text = extracted.text
        document.page_count = len(extracted.pages) if kind is DocumentKind.PDF else None
        document.pages = [
            DocumentPage(
                page_number=span.number, char_start=span.char_start, char_end=span.char_end
            )
            for span in extracted.page_spans()
        ]
        document.chunks.extend(build_chunks(extracted, index_version, embedder))
        document.status = DocumentStatus.COMPLETED
        document.error_code = document.error_message = None
        job.status = JobStatus.COMPLETED
        job.last_error = None
        job.finished_at = func.now()
        session.commit()
    except ExtractionError as exc:
        mark_failed(session, job, exc.code, str(exc))
    except TRANSIENT_ERRORS as exc:
        logger.warning(
            "Ingestion job %s: transient error on attempt %d of %d: %r",
            job_id,
            job.attempts,
            job.max_attempts,
            exc,
        )
        if job.attempts >= job.max_attempts:
            mark_failed(
                session, job, "retries_exhausted", f"Gave up after {job.attempts} attempts: {exc!r}"
            )
        else:
            mark_retrying(session, job, repr(exc))
    except IndexConfigurationError:
        # A deployment mistake, not the document's fault; the details go to the log only.
        logger.exception("Ingestion job %s: embedder does not match the index", job_id)
        mark_failed(session, job, "index_mismatch", "The worker cannot index documents right now.")
    except Exception:
        logger.exception("Unexpected error in ingestion job %s", job_id)
        mark_failed(session, job, "internal_error", "Unexpected error while processing.")
    return job.status


def ingest_upload(
    session: Session,
    blob_store: BlobStore,
    embedder: Embedder,
    *,
    owner_id: uuid.UUID,
    filename: str,
    data: bytes,
) -> UploadResult:
    """Accept and process in one call, without a queue (CLI ingest-dir and tests).

    A transient failure is retried straight away (no backoff: someone is waiting at the
    terminal); max_attempts still bounds the loop.
    """
    # Here the embedder is at hand, so a mismatch is refused before anything is stored.
    ensure_compatible(index_versions_repo.get_active(session), embedder)
    result = accept_upload(session, blob_store, owner_id=owner_id, filename=filename, data=data)
    if result.job is not None:
        # result.document is the same object the job updates, so it shows the outcome.
        while run_ingestion_job(session, blob_store, embedder, result.job.id) is JobStatus.RETRYING:
            pass
    return result


@dataclass(frozen=True)
class ReapResult:
    requeued: list[uuid.UUID]
    failed: list[uuid.UUID]


def reap_stale_jobs(session: Session, lease: timedelta) -> ReapResult:
    """Recover jobs whose worker died mid-job (killed, crashed, machine lost). Commits.

    Such a job stays "processing" forever: no worker owns it, and the claim refuses to take it.
    The crashed run already counted as an attempt, so a document that crashes the worker every
    time (e.g. runs it out of memory) ends up failed instead of looping. The caller enqueues
    the returned `requeued` jobs.
    """
    result = ReapResult(requeued=[], failed=[])
    for job in jobs_repo.stale(session, lease):
        message = "The worker stopped while processing this job."
        if job.attempts >= job.max_attempts:
            _set_failed(job, "worker_lost", message)
            result.failed.append(job.id)
        else:
            _set_retrying(job, message)
            result.requeued.append(job.id)
    session.commit()
    return result


def ensure_compatible(index_version: IndexVersion, embedder: Embedder) -> None:
    """Vectors from different models live in different spaces; mixing them breaks search."""
    if (index_version.embedding_model, index_version.embedding_dim) != (
        embedder.model_name,
        embedder.dim,
    ):
        raise IndexConfigurationError(
            f"Active index version {index_version.name!r} expects "
            f"{index_version.embedding_model} ({index_version.embedding_dim}d), but the loaded "
            f"embedder is {embedder.model_name} ({embedder.dim}d)."
        )


def build_chunks(
    extracted: ExtractedDocument, index_version: IndexVersion, embedder: Embedder
) -> list[Chunk]:
    """Chunk the document and embed every chunk, all in memory (nothing is committed)."""
    pieces = chunk_document(
        extracted, ChunkingConfig(**index_version.chunking), embedder.count_tokens
    )
    texts = [
        text_for_embedding(
            piece.text,
            title=extracted.title,
            section_path=piece.section_path,
            contextual_header=index_version.contextual_header,
            count_tokens=embedder.count_tokens,
            max_tokens=embedder.max_tokens,
        )
        for piece in pieces
    ]
    vectors = embedder.embed_documents(texts)
    return [
        Chunk(
            index_version_id=index_version.id,
            chunk_index=piece.index,
            text=piece.text,
            token_count=piece.token_count,
            char_start=piece.char_start,
            char_end=piece.char_end,
            page_start=piece.page_start,
            page_end=piece.page_end,
            section_path=piece.section_path,
            embedding=vector,
        )
        for piece, vector in zip(pieces, vectors, strict=True)
    ]


def stored_extraction(document: Document) -> ExtractedDocument:
    """Rebuild the extraction result from what the database already holds (no re-parsing)."""
    page_texts = (document.extracted_text or "").split(PAGE_SEPARATOR)
    if document.page_count:
        pages = tuple(PageText(number=i, text=t) for i, t in enumerate(page_texts, start=1))
    else:
        pages = (PageText(number=None, text=page_texts[0]),)
    return ExtractedDocument(title=document.title, pages=pages)


def reindex_document(session: Session, embedder: Embedder, document: Document) -> int:
    """(Re)build the chunks of a completed document for the active index version."""
    index_version = index_versions_repo.get_active(session)
    ensure_compatible(index_version, embedder)
    document.chunks = [c for c in document.chunks if c.index_version_id != index_version.id]
    session.flush()  # delete the old chunks before inserting new ones with the same indexes
    new_chunks = build_chunks(stored_extraction(document), index_version, embedder)
    document.chunks.extend(new_chunks)
    session.commit()
    return len(new_chunks)


def mark_failed(session: Session, job: IngestionJob, code: str, message: str) -> None:
    # Discard whatever the failed attempt changed, then record the failure in a fresh
    # transaction, so the document can never be left looking "processing" or "completed".
    session.rollback()
    _set_failed(job, code, message)
    session.commit()


def mark_retrying(session: Session, job: IngestionJob, error: str) -> None:
    session.rollback()  # the attempt's partial results are discarded, as for a failure
    _set_retrying(job, error)
    session.commit()


def _set_failed(job: IngestionJob, code: str, message: str) -> None:
    document = job.document
    document.status = DocumentStatus.FAILED
    document.error_code = code
    document.error_message = message[:1000]
    job.status = JobStatus.FAILED
    job.last_error = f"{code}: {message}"[:1000]
    job.finished_at = func.now()


def _set_retrying(job: IngestionJob, error: str) -> None:
    job.document.status = DocumentStatus.QUEUED  # waiting to be processed again
    job.status = JobStatus.RETRYING
    job.last_error = error[:1000]


def _safe_filename(filename: str) -> str:
    """Keep only the base name: the client controls this string, and some send full paths."""
    return PurePath(filename.replace("\\", "/")).name[:255] or "upload"
