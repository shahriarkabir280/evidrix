"""Failure modes of background ingestion: transient vs permanent errors, faults in the middle
of the final transaction, and jobs whose worker died (plan: Part 5, tests)."""

from datetime import timedelta

import pytest
import structlog
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import OperationalError

from app.db.models import Chunk, DocumentStatus, IngestionJob, JobStatus
from app.db.repositories import jobs as jobs_repo
from app.db.repositories.users import DEV_USER_ID
from app.services import ingestion
from app.services.ingestion import accept_upload, reap_stale_jobs, run_ingestion_job
from app.worker import jobs as worker_jobs
from tests.factories import make_pdf
from tests.fakes import FailingEmbedder

pytestmark = pytest.mark.integration

GUIDE = b"# Cookies\n\n" + b"Declare cookie parameters with the Cookie class. " * 20


def accept(session_factory, blob_store, data: bytes = GUIDE, filename: str = "cookies.md"):
    with session_factory() as session:
        job = accept_upload(
            session, blob_store, owner_id=DEV_USER_ID, filename=filename, data=data
        ).job
    assert job is not None
    return job.id


def run(session_factory, blob_store, embedder, job_id) -> JobStatus | None:
    with session_factory() as session:
        return run_ingestion_job(session, blob_store, embedder, job_id)


def load(session_factory, job_id) -> tuple[IngestionJob, int]:
    """The job (with its document loaded) and the number of chunks in the database."""
    with session_factory() as session:
        job = session.get_one(IngestionJob, job_id)
        job.document  # noqa: B018 (load it while the session is open)
        return job, session.scalar(select(func.count()).select_from(Chunk))


# --- Transient errors are retried, up to max_attempts -------------------------------------


def test_embedding_timeout_marks_the_job_retrying(session_factory, blob_store):
    job_id = accept(session_factory, blob_store)

    status = run(
        session_factory, blob_store, FailingEmbedder(TimeoutError("embed timeout")), job_id
    )

    job, chunks = load(session_factory, job_id)
    assert status is JobStatus.RETRYING
    assert (job.status, job.attempts) == (JobStatus.RETRYING, 1)
    assert "embed timeout" in job.last_error
    assert job.document.status is DocumentStatus.QUEUED
    assert chunks == 0


def test_a_retry_after_a_transient_error_succeeds(session_factory, blob_store):
    job_id = accept(session_factory, blob_store)
    embedder = FailingEmbedder(ConnectionError("refused"), failures=1)

    statuses = [run(session_factory, blob_store, embedder, job_id) for _ in range(2)]

    job, chunks = load(session_factory, job_id)
    assert statuses == [JobStatus.RETRYING, JobStatus.COMPLETED]
    assert (job.attempts, job.last_error) == (2, None)
    assert job.document.status is DocumentStatus.COMPLETED
    assert chunks > 0


def test_embedding_provider_down_fails_after_max_attempts_with_zero_chunks(
    session_factory, blob_store
):
    job_id = accept(session_factory, blob_store)
    embedder = FailingEmbedder(TimeoutError("embed timeout"))

    statuses = [run(session_factory, blob_store, embedder, job_id) for _ in range(5)]

    job, chunks = load(session_factory, job_id)
    assert statuses == [JobStatus.RETRYING] * 3 + [JobStatus.FAILED, None]
    assert (job.status, job.attempts) == (JobStatus.FAILED, 4)
    assert job.document.status is DocumentStatus.FAILED
    assert job.document.error_code == "retries_exhausted"
    assert chunks == 0


# --- Permanent errors fail at once --------------------------------------------------------


def test_extraction_error_is_not_retried(session_factory, blob_store, embedder):
    job_id = accept(session_factory, blob_store, b"%PDF-1.7\nnot a pdf", "broken.pdf")

    status = run(session_factory, blob_store, embedder, job_id)

    job, _ = load(session_factory, job_id)
    assert status is JobStatus.FAILED
    assert (job.attempts, job.document.error_code) == (1, "invalid_pdf")


def test_unexpected_error_is_not_retried(session_factory, blob_store, embedder, monkeypatch):
    def bug(*args, **kwargs):
        raise KeyError("bug in the chunker")

    monkeypatch.setattr(ingestion, "build_chunks", bug)
    job_id = accept(session_factory, blob_store)

    status = run(session_factory, blob_store, embedder, job_id)

    job, _ = load(session_factory, job_id)
    assert status is JobStatus.FAILED
    assert job.document.error_code == "internal_error"


# --- A fault inside the final transaction leaves nothing behind ---------------------------


def test_fault_after_chunks_are_written_rolls_back_and_the_retry_has_no_duplicates(
    session_factory, blob_store, embedder
):
    pdf = make_pdf(["First page about cookies and headers.", "Second page about responses."])
    job_id = accept(session_factory, blob_store, pdf, "guide.pdf")
    seen_in_transaction = []

    with session_factory() as session:

        @event.listens_for(session, "before_commit")
        def fail_once_chunks_are_written(session):
            if seen_in_transaction:
                return
            session.flush()
            chunks = session.scalar(select(func.count()).select_from(Chunk))
            if chunks:  # the chunks and pages are written, but not committed yet
                seen_in_transaction.append(chunks)
                raise OperationalError("COMMIT", {}, Exception("server closed the connection"))

        first = run_ingestion_job(session, blob_store, embedder, job_id)

    with session_factory() as session:
        after_fault = session.scalar(select(func.count()).select_from(Chunk))
        pages_after_fault = session.scalar(text("SELECT count(*) FROM document_pages"))
    second = run(session_factory, blob_store, embedder, job_id)
    _, chunks = load(session_factory, job_id)

    assert first is JobStatus.RETRYING
    assert seen_in_transaction and seen_in_transaction[0] > 0
    assert (after_fault, pages_after_fault) == (0, 0)
    assert second is JobStatus.COMPLETED
    assert chunks == seen_in_transaction[0]


# --- Jobs whose worker died are recovered -------------------------------------------------


def claim_and_age(session_factory, job_id, *, minutes: int, attempts: int | None = None):
    """Claim the job, then pretend the worker died `minutes` ago without finishing."""
    with session_factory() as session:
        jobs_repo.claim(session, job_id)
        session.execute(
            text(
                "UPDATE ingestion_jobs SET started_at = now() - make_interval(mins => :m), "
                "attempts = coalesce(:attempts, attempts) WHERE id = :id"
            ),
            {"m": minutes, "attempts": attempts, "id": job_id},
        )
        session.commit()


def test_stale_job_is_requeued_and_can_then_be_processed(session_factory, blob_store, embedder):
    job_id = accept(session_factory, blob_store)
    claim_and_age(session_factory, job_id, minutes=20)

    with session_factory() as session:
        reaped = reap_stale_jobs(session, timedelta(minutes=15))
    status = run(session_factory, blob_store, embedder, job_id)

    job, chunks = load(session_factory, job_id)
    assert reaped.requeued == [job_id]
    assert status is JobStatus.COMPLETED
    assert job.attempts == 2  # the crashed run counted as an attempt
    assert chunks > 0


def test_job_that_keeps_killing_its_worker_ends_up_failed(session_factory, blob_store):
    job_id = accept(session_factory, blob_store)
    claim_and_age(session_factory, job_id, minutes=20, attempts=4)

    with session_factory() as session:
        reaped = reap_stale_jobs(session, timedelta(minutes=15))

    job, _ = load(session_factory, job_id)
    assert (reaped.requeued, reaped.failed) == ([], [job_id])
    assert job.status is JobStatus.FAILED
    assert job.document.error_code == "worker_lost"


def test_job_within_its_lease_is_left_alone(session_factory, blob_store):
    job_id = accept(session_factory, blob_store)
    claim_and_age(session_factory, job_id, minutes=5)

    with session_factory() as session:
        reaped = reap_stale_jobs(session, timedelta(minutes=15))

    job, _ = load(session_factory, job_id)
    assert reaped.requeued == reaped.failed == []
    assert job.status is JobStatus.PROCESSING


# --- The RQ entry point turns "retrying" into a failed RQ run -----------------------------


def test_worker_raises_so_that_rq_schedules_the_retry(monkeypatch):
    monkeypatch.setattr(worker_jobs, "run_ingestion_job", lambda *a: JobStatus.RETRYING)
    monkeypatch.setattr(worker_jobs, "get_embedder", lambda: None)

    with pytest.raises(worker_jobs.JobWillRetryError):
        worker_jobs.process_ingestion("00000000-0000-0000-0000-000000000000")


def test_worker_returns_the_final_status(monkeypatch):
    monkeypatch.setattr(worker_jobs, "run_ingestion_job", lambda *a: JobStatus.COMPLETED)
    monkeypatch.setattr(worker_jobs, "get_embedder", lambda: None)

    assert worker_jobs.process_ingestion("00000000-0000-0000-0000-000000000000") == "completed"


def test_worker_leaves_no_job_context_behind(monkeypatch):
    def bind_like_a_job(*args):
        structlog.contextvars.bind_contextvars(request_id="from-the-api", job_id="job-1")
        return JobStatus.COMPLETED

    monkeypatch.setattr(worker_jobs, "run_ingestion_job", bind_like_a_job)
    monkeypatch.setattr(worker_jobs, "get_embedder", lambda: None)

    worker_jobs.process_ingestion("00000000-0000-0000-0000-000000000000")

    assert structlog.contextvars.get_contextvars() == {}
