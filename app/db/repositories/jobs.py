"""Ingestion job queries, including the atomic claim.

Component 5A/5B.
"""

import uuid
from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import Document, IngestionJob, JobStatus

CLAIMABLE = (JobStatus.QUEUED, JobStatus.RETRYING)


def claim(session: Session, job_id: uuid.UUID) -> IngestionJob | None:
    """Mark the job as processing if, and only if, it is waiting to run. Commits.

    One UPDATE both checks and changes the status, so two workers that receive the same job
    can't both pass the check: Postgres locks the row, and the second UPDATE sees 'processing'
    and matches nothing. None means a duplicate delivery or a job that is already done.
    """
    job = session.scalars(
        update(IngestionJob)
        .where(IngestionJob.id == job_id, IngestionJob.status.in_(CLAIMABLE))
        .values(
            status=JobStatus.PROCESSING,
            attempts=IngestionJob.attempts + 1,
            started_at=func.now(),
        )
        .returning(IngestionJob)
        # Overwrite the copy this session may already hold with the row as it is now.
        .execution_options(populate_existing=True)
    ).one_or_none()
    # Commit the claim on its own, so it holds even if processing crashes the worker.
    session.commit()
    return job


def stale(session: Session, older_than: timedelta) -> list[IngestionJob]:
    """Jobs that have been processing for longer than `older_than`, locked for update.

    SKIP LOCKED leaves out a job whose row is locked right now, i.e. one a live worker is
    finishing at this very moment.
    """
    return list(
        session.scalars(
            select(IngestionJob)
            .where(
                IngestionJob.status == JobStatus.PROCESSING,
                IngestionJob.started_at < func.now() - older_than,
            )
            .with_for_update(skip_locked=True)
        )
    )


def get_for_owner(session: Session, owner_id: uuid.UUID, job_id: uuid.UUID) -> IngestionJob | None:
    return session.scalar(
        select(IngestionJob)
        .join(Document, IngestionJob.document_id == Document.id)
        .where(IngestionJob.id == job_id, Document.owner_id == owner_id)
    )


def latest_for_document(session: Session, document_id: uuid.UUID) -> IngestionJob | None:
    return session.scalar(
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.enqueued_at.desc())
        .limit(1)
    )
