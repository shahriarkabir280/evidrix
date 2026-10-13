"""RQ job functions: process_ingestion(job_id).

Component 5A/5B.

    uv run python -m app.cli reap-jobs
    uv run rq worker --worker-class rq.worker.SimpleWorker --with-scheduler ingestion

SimpleWorker runs jobs in its own process instead of forking a child per job. Forking a process
that has loaded torch can crash on macOS, and without forking the embedding model stays loaded
between jobs. --with-scheduler lets the worker run the delayed retries.
"""

import uuid

import structlog

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import JobStatus
from app.db.session import get_session_factory
from app.rag.embeddings import get_embedder
from app.services.ingestion import run_ingestion_job
from app.storage.blob import make_blob_store

# The worker process imports this module before running its first job.
configure_logging(get_settings())


class JobWillRetryError(Exception):
    """Raised so that RQ schedules the next attempt (the Retry given at enqueue time)."""


def process_ingestion(job_id: str) -> str | None:
    # One worker process runs many jobs. The context (request ID, job ID) is cleared before and
    # after each one; otherwise RQ's lines about the next job would carry this job's request ID.
    structlog.contextvars.clear_contextvars()
    settings = get_settings()
    try:
        with get_session_factory()() as session:
            status: JobStatus | None = run_ingestion_job(
                session, make_blob_store(settings), get_embedder(), uuid.UUID(job_id)
            )
    finally:
        structlog.contextvars.clear_contextvars()
    if status is JobStatus.RETRYING:
        # The job row already says "retrying" and holds the error; RQ only needs to know
        # that this run failed, to schedule the next one with backoff.
        raise JobWillRetryError(f"Ingestion job {job_id} hit a transient error")
    return status.value if status else None
