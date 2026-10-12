"""Structured logs and request IDs (component 5G): every line carries the request ID, and no
line carries a key, a question or document text."""

import io
import json

import pytest
from sqlalchemy import select

from app.api.deps import get_llm
from app.core.config import Settings
from app.core.logging import configure_logging
from app.db.models import IngestionJob
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration

SECRET_TEXT = "quartzflamingo9182"
DOC = f"# Notes\n\nThe internal codename of the release is {SECRET_TEXT}.\n".encode()
QUESTION = "What is the internal codename of the release?"


@pytest.fixture
def logs(client):
    """The JSON lines logged during the test (after the app's own startup configured logging)."""
    buffer = io.StringIO()
    configure_logging(Settings(_env_file=None), stream=buffer)
    return lambda: [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_every_response_has_a_request_id(client):
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]

    assert len(first) == 32 and first != second


def test_a_safe_client_request_id_is_kept(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})

    assert response.headers["X-Request-ID"] == "trace-abc.123"


@pytest.mark.parametrize("supplied", ["has spaces", "x" * 65, 'quote"}{', "tab\there"])
def test_an_unsafe_client_request_id_is_replaced(client, supplied):
    response = client.get("/health", headers={"X-Request-ID": supplied})

    assert response.headers["X-Request-ID"] != supplied
    assert len(response.headers["X-Request-ID"]) == 32


def test_access_line_has_request_id_status_and_user(client, logs):
    response = client.get("/documents")

    access = [line for line in logs() if line["event"] == "request"]
    assert access == [
        {
            "event": "request",
            "method": "GET",
            "path": "/documents",
            "status": 200,
            "duration_ms": access[0]["duration_ms"],
            "user_id": "00000000-0000-0000-0000-000000000001",
            "request_id": response.headers["X-Request-ID"],
            "level": "info",
            "logger": "app.access",
            "timestamp": access[0]["timestamp"],
        }
    ]


def test_worker_lines_carry_the_request_id_of_the_upload(client, run_worker, db_session, logs):
    response = client.post("/documents", files={"file": ("notes.md", DOC)})
    request_id = response.headers["X-Request-ID"]
    run_worker()

    job = db_session.scalars(select(IngestionJob)).one()
    worker = [line for line in logs() if line["event"].startswith("ingestion_job")]
    assert job.request_id == request_id
    assert [line["event"] for line in worker] == [
        "ingestion_job_started",
        "ingestion_job_completed",
    ]
    assert {line["request_id"] for line in worker} == {request_id}
    assert {line["job_id"] for line in worker} == {str(job.id)}


def test_no_log_line_contains_a_key_a_question_or_document_text(client, ingest, api_key, logs):
    reply = f'{{"answer": "It is {SECRET_TEXT} [1].", "citations": [1], "abstain": false}}'
    client.app.dependency_overrides[get_llm] = lambda: FakeLLM(reply)

    ingest("notes.md", DOC)
    answer = client.post("/chat", json={"question": QUESTION}).json()["answer"]
    client.get("/documents", headers={"Authorization": "Bearer evx_" + "B" * 43})

    lines = logs()
    output = json.dumps(lines)
    chat = [line for line in lines if line["event"] == "chat_answered"]
    assert SECRET_TEXT in answer  # the secret did flow through the system...
    assert SECRET_TEXT not in output  # ...but not into the logs
    assert QUESTION not in output
    assert api_key not in output and "evx_" not in output
    assert chat[0]["question"]["chars"] == len(QUESTION)
