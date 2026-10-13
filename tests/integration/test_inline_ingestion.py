"""INGESTION_MODE=inline: the API process runs the jobs itself (component 7B)."""

import pytest

from app.core.config import Settings
from app.db.models import IngestionJob, JobStatus
from app.db.repositories import jobs as jobs_repo
from app.db.repositories.users import DEV_USER_ID
from app.services.ingestion import accept_upload
from app.worker import inline
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import FailingEmbedder

pytestmark = pytest.mark.integration

GUIDE = b"# Cookies\n\n" + b"Declare cookie parameters with the Cookie class. " * 20


@pytest.fixture
def make_queue(monkeypatch, session_factory, blob_store, embedder):
    """An inline queue wired to the test database, blob store and embedder."""

    def make(*, embedder_override=None, delays=(0, 0, 0)) -> inline.InlineIngestionQueue:
        monkeypatch.setattr(inline, "get_session_factory", lambda: session_factory)
        monkeypatch.setattr(inline, "make_blob_store", lambda settings: blob_store)
        monkeypatch.setattr(inline, "get_embedder", lambda: embedder_override or embedder)
        settings = Settings(
            _env_file=None,
            database_url=TEST_DATABASE_URL,
            ingestion_mode="inline",
            ingestion_retry_delays_seconds=list(delays),
        )
        return inline.InlineIngestionQueue(settings)

    return make


def accept(session_factory, blob_store, data: bytes = GUIDE):
    with session_factory() as session:
        job = accept_upload(
            session, blob_store, owner_id=DEV_USER_ID, filename="cookies.md", data=data
        ).job
    return job.id


def job_state(session_factory, job_id) -> tuple[JobStatus, int]:
    with session_factory() as session:
        job = session.get_one(IngestionJob, job_id)
        return job.status, job.attempts


def drain(queue: inline.InlineIngestionQueue) -> None:
    queue._executor.shutdown(wait=True)


def test_enqueued_job_runs_in_the_background(make_queue, session_factory, blob_store):
    queue = make_queue()
    job_id = accept(session_factory, blob_store)

    queue.enqueue(job_id)
    drain(queue)

    assert job_state(session_factory, job_id) == (JobStatus.COMPLETED, 1)


def test_transient_errors_are_retried_in_the_background(make_queue, session_factory, blob_store):
    queue = make_queue(embedder_override=FailingEmbedder(TimeoutError("slow"), failures=2))
    job_id = accept(session_factory, blob_store)

    queue.enqueue(job_id)
    drain(queue)

    assert job_state(session_factory, job_id) == (JobStatus.COMPLETED, 3)


def test_startup_recovers_jobs_the_last_process_left_behind(
    make_queue, session_factory, blob_store
):
    never_started = accept(session_factory, blob_store)
    interrupted = accept(session_factory, blob_store, GUIDE + b"Another file.")
    with session_factory() as session:
        jobs_repo.claim(session, interrupted)  # "processing" when the old process died

    queue = make_queue()
    resubmitted = queue.recover()
    drain(queue)

    assert resubmitted == 2
    assert job_state(session_factory, never_started) == (JobStatus.COMPLETED, 1)
    assert job_state(session_factory, interrupted) == (JobStatus.COMPLETED, 2)


def test_upload_through_the_api_is_processed_inline(client, make_queue, session_factory):
    from app.api.deps import get_ingestion_queue

    queue = make_queue()
    client.app.dependency_overrides[get_ingestion_queue] = lambda: queue

    response = client.post("/documents", files={"file": ("cookies.md", GUIDE)})
    drain(queue)

    assert response.status_code == 202
    assert client.get(f"/documents/{response.json()['id']}").json()["status"] == "completed"


def test_chat_can_be_switched_off_for_a_retrieval_only_deployment(client):
    from app.api.deps import get_settings

    client.app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, database_url=TEST_DATABASE_URL, chat_enabled=False
    )

    chat = client.post("/chat", json={"question": "How do I declare cookies?"})
    retrieve = client.post("/retrieve", json={"question": "How do I declare cookies?"})

    assert chat.status_code == 503
    assert chat.json()["error"]["code"] == "chat_disabled"
    assert retrieve.status_code == 200
