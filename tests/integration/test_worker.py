import pytest
from sqlalchemy import func, select

from app.db.models import Chunk, DocumentStatus, IngestionJob, JobStatus
from app.db.repositories import jobs as jobs_repo
from app.db.repositories.users import DEV_USER_ID
from app.services.ingestion import accept_upload, run_ingestion_job

pytestmark = pytest.mark.integration

GUIDE = b"# Cookies\n\n" + b"Declare cookie parameters with the Cookie class. " * 20


def accept(session, blob_store) -> IngestionJob:
    result = accept_upload(
        session, blob_store, owner_id=DEV_USER_ID, filename="cookies.md", data=GUIDE
    )
    assert result.job is not None
    return result.job


def chunk_count(session) -> int:
    return session.scalar(select(func.count()).select_from(Chunk))


def test_accepted_upload_is_queued_with_no_chunks(session_factory, blob_store):
    with session_factory() as session:
        job = accept(session, blob_store)

        assert job.status is JobStatus.QUEUED
        assert job.document.status is DocumentStatus.QUEUED
        assert chunk_count(session) == 0


def test_worker_processes_a_queued_job(session_factory, blob_store, embedder):
    with session_factory() as session:
        job = accept(session, blob_store)

    with session_factory() as session:
        status = run_ingestion_job(session, blob_store, embedder, job.id)
        job = session.get_one(IngestionJob, job.id)

        assert status is JobStatus.COMPLETED
        assert job.document.status is DocumentStatus.COMPLETED
        assert job.attempts == 1
        assert chunk_count(session) > 0


def test_repeated_delivery_is_a_no_op(session_factory, blob_store, embedder):
    with session_factory() as session:
        job = accept(session, blob_store)
    with session_factory() as session:
        run_ingestion_job(session, blob_store, embedder, job.id)
        chunks_after_first = chunk_count(session)

    with session_factory() as session:
        second = run_ingestion_job(session, blob_store, embedder, job.id)

        assert second is None
        assert chunk_count(session) == chunks_after_first
        assert session.get_one(IngestionJob, job.id).attempts == 1


def test_only_one_of_two_claims_wins(session_factory, blob_store):
    with session_factory() as session:
        job = accept(session, blob_store)

    with session_factory() as first, session_factory() as second:
        claims = [jobs_repo.claim(first, job.id), jobs_repo.claim(second, job.id)]

    assert claims[0] is not None and claims[0].status is JobStatus.PROCESSING
    assert claims[1] is None
