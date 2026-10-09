import pytest
from sqlalchemy import func, select

from app.api.deps import DEV_USER_ID
from app.db.models import Document, DocumentPage, DocumentStatus
from app.db.repositories import documents as documents_repo
from app.services import ingestion
from tests.factories import make_pdf

pytestmark = pytest.mark.integration


def test_concurrent_duplicate_upload_returns_the_winning_document(
    session_factory, blob_store, monkeypatch
):
    data = b"# Race\n\nTwo requests upload this exact file at the same moment."
    with session_factory() as session:
        winner = ingestion.ingest_upload(
            session, blob_store, owner_id=DEV_USER_ID, filename="race.md", data=data
        ).document

    # Simulate the race: the second request checked for a duplicate before the first
    # committed, so its lookup saw nothing and it goes on to insert the same file.
    real_lookup = documents_repo.get_by_hash
    calls = []

    def lookup_misses_once(*args, **kwargs):
        calls.append(1)
        return None if len(calls) == 1 else real_lookup(*args, **kwargs)

    monkeypatch.setattr(documents_repo, "get_by_hash", lookup_misses_once)

    with session_factory() as session:
        result = ingestion.ingest_upload(
            session, blob_store, owner_id=DEV_USER_ID, filename="race.md", data=data
        )
        count = session.scalar(select(func.count()).select_from(Document))

    assert result.created is False
    assert result.document.id == winner.id
    assert count == 1


def test_unexpected_error_marks_document_failed_without_partial_pages(
    session_factory, blob_store, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("bug in the extractor")

    monkeypatch.setattr(ingestion, "extract", explode)
    data = make_pdf(["Some page text that will never be extracted."])

    with session_factory() as session:
        result = ingestion.ingest_upload(
            session, blob_store, owner_id=DEV_USER_ID, filename="boom.pdf", data=data
        )
        page_count = session.scalar(select(func.count()).select_from(DocumentPage))

    assert result.document.status is DocumentStatus.FAILED
    assert result.document.error_code == "internal_error"
    assert page_count == 0


def test_status_is_processing_while_extraction_runs(session_factory, blob_store, monkeypatch):
    seen_statuses = []
    real_extract = ingestion.extract

    def spy(*args, **kwargs):
        # Look at the row from a separate connection, as another request would.
        with session_factory() as other:
            seen_statuses.append(other.scalar(select(Document.status)))
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(ingestion, "extract", spy)

    with session_factory() as session:
        result = ingestion.ingest_upload(
            session,
            blob_store,
            owner_id=DEV_USER_ID,
            filename="notes.md",
            data=b"# Notes\n\nSome notes about background tasks.",
        )

    assert seen_statuses == [DocumentStatus.PROCESSING]
    assert result.document.status is DocumentStatus.COMPLETED
