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

from app.db.models import Chunk, Document, DocumentPage, DocumentStatus, IndexVersion
from app.db.repositories import documents as documents_repo
from app.db.repositories import index_versions as index_versions_repo
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


class IndexConfigurationError(Exception):
    """The running embedder doesn't match the active index version (a deployment mistake)."""


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
    embedder: Embedder,
    *,
    owner_id: uuid.UUID,
    filename: str,
    data: bytes,
) -> IngestResult:
    filename = _safe_filename(filename)
    kind = detect_kind(filename, data)
    content_hash = content_key(data)
    index_version = index_versions_repo.get_active(session)
    ensure_compatible(index_version, embedder)

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
        document.chunks = build_chunks(extracted, index_version, embedder)
        document.status = DocumentStatus.COMPLETED
        session.commit()
    except ExtractionError as exc:
        _mark_failed(session, document, exc.code, str(exc))
    except Exception:
        logger.exception("Unexpected error while ingesting document %s", document.id)
        _mark_failed(session, document, "internal_error", "Unexpected error while processing.")
    return IngestResult(document, created=True)


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
