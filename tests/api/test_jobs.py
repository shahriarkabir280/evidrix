import uuid

import pytest

from app.db.models import Document, IngestionJob
from app.db.repositories import index_versions as index_versions_repo

pytestmark = pytest.mark.integration

MARKDOWN = b"# Background tasks\n\nRun work after returning a response.\n"


def upload(client, filename: str, data: bytes) -> dict:
    return client.post("/documents", files={"file": (filename, data)}).json()


def test_job_moves_from_queued_to_completed(client, run_worker):
    job_id = upload(client, "tasks.md", MARKDOWN)["job_id"]

    queued = client.get(f"/jobs/{job_id}").json()
    run_worker()
    done = client.get(f"/jobs/{job_id}").json()

    assert (queued["status"], queued["attempts"], queued["started_at"]) == ("queued", 0, None)
    assert (done["status"], done["attempts"]) == ("completed", 1)
    assert done["started_at"] and done["finished_at"]
    assert done["last_error"] is None


def test_failed_job_reports_the_error(client, run_worker):
    job_id = upload(client, "broken.pdf", b"%PDF-1.7\nnot really a pdf body")["job_id"]

    run_worker()
    job = client.get(f"/jobs/{job_id}").json()

    assert job["status"] == "failed"
    assert job["last_error"].startswith("invalid_pdf:")
    assert job["finished_at"]


def test_unknown_job_is_not_found(client):
    response = client.get(f"/jobs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "job_not_found"


def test_other_users_jobs_are_invisible(client, db_session, other_user):
    document = Document(
        owner_id=other_user.id,
        title="Private",
        filename="private.md",
        mime_type="text/markdown",
        size_bytes=10,
        content_hash="e" * 64,
        storage_key="e" * 64,
    )
    job = IngestionJob(
        document=document,
        index_version_id=index_versions_repo.get_active(db_session).id,
        max_attempts=4,
    )
    db_session.add_all([document, job])
    db_session.commit()

    assert client.get(f"/jobs/{job.id}").status_code == 404
