"""Hands ingestion jobs to the worker through Redis (RQ).

Component 5A · retries in 5B.
"""

import uuid
from functools import lru_cache
from typing import Protocol

from redis import Redis, RedisError
from rq import Queue, Retry

from app.core.config import get_settings

QUEUE_NAME = "ingestion"


class QueueUnavailableError(Exception):
    """The job could not be handed to the queue (Redis is down or unreachable)."""


class IngestionQueue(Protocol):
    def enqueue(self, job_id: uuid.UUID) -> None: ...


class RQIngestionQueue:
    def __init__(
        self,
        redis_url: str,
        *,
        job_timeout_seconds: int,
        max_attempts: int,
        retry_delays_seconds: list[int],
    ) -> None:
        self._queue = Queue(QUEUE_NAME, connection=Redis.from_url(redis_url))
        self._job_timeout = job_timeout_seconds
        # RQ counts retries, the job row counts attempts: 4 attempts = the first run + 3 retries.
        # Delayed retries need a worker started with --with-scheduler.
        self._retry = Retry(max=max_attempts - 1, interval=retry_delays_seconds)

    def enqueue(self, job_id: uuid.UUID) -> None:
        # The function is named by its import path: the API never imports the worker code, and
        # the message carries only the job id (the worker reads the rest from Postgres).
        try:
            self._queue.enqueue_call(
                "app.worker.jobs.process_ingestion",
                args=(str(job_id),),
                job_id=str(job_id),
                timeout=self._job_timeout,
                retry=self._retry,
            )
        except RedisError as exc:
            raise QueueUnavailableError(str(exc)) from exc


@lru_cache
def get_ingestion_queue() -> IngestionQueue:
    settings = get_settings()
    if settings.ingestion_mode == "inline":
        from app.worker.inline import InlineIngestionQueue

        return InlineIngestionQueue(settings)
    return RQIngestionQueue(
        settings.redis_url,
        job_timeout_seconds=settings.ingestion_job_timeout_seconds,
        max_attempts=settings.ingestion_max_attempts,
        retry_delays_seconds=settings.ingestion_retry_delays_seconds,
    )
