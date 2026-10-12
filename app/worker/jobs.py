"""RQ job functions: process_ingestion(job_id).

Component 5A/5B.

    uv run rq worker --worker-class rq.worker.SimpleWorker ingestion

SimpleWorker runs jobs in its own process instead of forking a child per job. Forking a process
that has loaded torch can crash on macOS, and without forking the embedding model stays loaded
between jobs.
"""

import uuid

from app.core.config import get_settings
from app.db.models import JobStatus
from app.db.session import get_session_factory
from app.rag.embeddings import get_embedder
from app.services.ingestion import run_ingestion_job
from app.storage.blob import LocalBlobStore


def process_ingestion(job_id: str) -> str | None:
    settings = get_settings()
    with get_session_factory()() as session:
        status: JobStatus | None = run_ingestion_job(
            session, LocalBlobStore(settings.blob_dir), get_embedder(), uuid.UUID(job_id)
        )
    return status.value if status else None
