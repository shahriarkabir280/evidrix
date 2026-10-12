"""GET /jobs/{id}: ingestion job status.

Component 5A.
"""

import uuid

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUserId, DbSession
from app.api.schemas import JobOut
from app.db.repositories import jobs as jobs_repo

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: uuid.UUID, session: DbSession, owner_id: CurrentUserId) -> JobOut:
    job = jobs_repo.get_for_owner(session, owner_id, job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, {"code": "job_not_found", "message": "Job not found."}
        )
    return JobOut.model_validate(job)
