"""Ingestion inside the API process, for a deployment without a separate worker.

Component 7B · INGESTION_MODE=inline.

A one-thread executor plays the worker: jobs run one at a time after the upload's response is
sent, so memory stays bounded on a small instance. The trade-offs against the RQ worker:
- **Durability:** a job lives only in this process's memory between enqueue and start. The job
  row is still committed first, so `recover()` finds and resubmits unfinished jobs at startup.
  That is only safe with a single API instance: a second one would take over jobs the first is
  still running.
- **Retries:** the same attempts and delays as RQ, but slept in the background thread.
- **Load:** ingestion competes with requests for the CPU (the models are serialized by
  INFERENCE_LOCK anyway).
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import structlog
from sqlalchemy import select

from app.core.config import Settings
from app.db.models import IngestionJob, JobStatus
from app.db.session import get_session_factory
from app.rag.embeddings import get_embedder
from app.services.ingestion import reap_stale_jobs, run_ingestion_job
from app.storage.blob import make_blob_store

logger = structlog.get_logger(__name__)


class InlineIngestionQueue:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingestion")

    def enqueue(self, job_id: uuid.UUID) -> None:
        self._executor.submit(self._run, job_id)

    def recover(self) -> int:
        """Resubmit jobs a previous process left queued, retrying or processing. Returns how many.

        Called once at startup, before any request: in a single process nothing else can be
        running them, so "processing" means the process that ran them is gone.
        """
        with get_session_factory()() as session:
            reaped = reap_stale_jobs(session, timedelta(0))
            waiting = session.scalars(
                select(IngestionJob.id).where(
                    IngestionJob.status.in_([JobStatus.QUEUED, JobStatus.RETRYING])
                )
            ).all()
        for job_id in waiting:
            self.enqueue(job_id)
        logger.info(
            "inline_ingestion_recovered", resubmitted=len(waiting), failed=len(reaped.failed)
        )
        return len(waiting)

    def _run(self, job_id: uuid.UUID) -> None:
        structlog.contextvars.clear_contextvars()
        delays = list(self._settings.ingestion_retry_delays_seconds)
        try:
            while True:
                with get_session_factory()() as session:
                    status = run_ingestion_job(
                        session, make_blob_store(self._settings), get_embedder(), job_id
                    )
                if status is not JobStatus.RETRYING:
                    return
                time.sleep(delays.pop(0) if delays else 60)
        except Exception:
            # Nothing above this thread would log it; the job row may stay "processing" until
            # the next startup's recover().
            logger.exception("inline_ingestion_crashed", job_id=str(job_id))
        finally:
            structlog.contextvars.clear_contextvars()
