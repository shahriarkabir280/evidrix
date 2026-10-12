"""Hands ingestion jobs to the worker through Redis (RQ).

Component 5A.
"""

import uuid
from functools import lru_cache
from typing import Protocol

from redis import Redis, RedisError
from rq import Queue

from app.core.config import get_settings

QUEUE_NAME = "ingestion"


class QueueUnavailableError(Exception):
    """The job could not be handed to the queue (Redis is down or unreachable)."""


class IngestionQueue(Protocol):
    def enqueue(self, job_id: uuid.UUID) -> None: ...


class RQIngestionQueue:
    def __init__(self, redis_url: str, job_timeout_seconds: int) -> None:
        self._queue = Queue(QUEUE_NAME, connection=Redis.from_url(redis_url))
        self._job_timeout = job_timeout_seconds

    def enqueue(self, job_id: uuid.UUID) -> None:
        # The function is named by its import path: the API never imports the worker code, and
        # the message carries only the job id (the worker reads the rest from Postgres).
        try:
            self._queue.enqueue_call(
                "app.worker.jobs.process_ingestion",
                args=(str(job_id),),
                job_id=str(job_id),
                timeout=self._job_timeout,
            )
        except RedisError as exc:
            raise QueueUnavailableError(str(exc)) from exc


@lru_cache
def get_ingestion_queue() -> IngestionQueue:
    settings = get_settings()
    return RQIngestionQueue(settings.redis_url, settings.ingestion_job_timeout_seconds)
