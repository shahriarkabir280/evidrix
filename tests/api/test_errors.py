"""The error envelope (component 6C): one shape for every 4xx and 5xx."""

import uuid

import pytest
from redis import RedisError

from app.api.deps import get_db, get_redis, get_settings
from app.core.config import Settings
from tests.conftest import TEST_DATABASE_URL, TEST_MAX_UPLOAD_BYTES

pytestmark = pytest.mark.integration


def assert_envelope(response, status_code: int, code: str) -> dict:
    assert response.status_code == status_code, response.text
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str)
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


def test_401_missing_key(client):
    client.headers.pop("Authorization")
    assert_envelope(client.get("/documents"), 401, "invalid_api_key")


def test_404_unknown_document(client):
    assert_envelope(client.get(f"/documents/{uuid.uuid4()}"), 404, "document_not_found")


def test_404_unknown_path(client):
    assert_envelope(client.get("/no-such-thing"), 404, "not_found")


def test_405_wrong_method(client):
    assert_envelope(client.delete("/health"), 405, "method_not_allowed")


def test_409_document_still_processing(client):
    document_id = client.post("/documents", files={"file": ("a.md", b"# A\n\nSome text.")}).json()[
        "id"
    ]  # queued, the worker hasn't run

    response = client.post("/chat", json={"question": "q", "document_ids": [document_id]})

    assert_envelope(response, 409, "document_not_ready")


def test_413_too_large(client):
    response = client.post(
        "/documents", files={"file": ("big.txt", b"a" * (TEST_MAX_UPLOAD_BYTES + 1))}
    )
    assert_envelope(response, 413, "file_too_large")


def test_415_unsupported_type(client):
    response = client.post("/documents", files={"file": ("tool.exe", b"MZ")})
    assert_envelope(response, 415, "unsupported_file_type")


def test_422_validation_lists_each_problem(client):
    response = client.post("/chat", json={"question": "", "retrieval": {"top_k": "many"}})

    error = assert_envelope(response, 422, "validation_error")
    fields = {detail["field"] for detail in error["details"]}
    assert {"body.question", "body.retrieval.top_k"} <= fields


def test_429_keeps_retry_after(client):
    for _ in range(60):
        client.post("/retrieve", json={"question": "q"})

    response = client.post("/retrieve", json={"question": "q"})

    assert_envelope(response, 429, "rate_limited")
    assert int(response.headers["Retry-After"]) >= 1


def test_500_returns_only_the_request_id(client):
    def broken_settings():
        raise RuntimeError("secret detail: password=hunter2 at /srv/app/config.py")

    client.app.dependency_overrides[get_settings] = broken_settings

    response = client.post("/chat", json={"question": "q"})

    error = assert_envelope(response, 500, "internal_error")
    assert "hunter2" not in response.text and "config.py" not in response.text
    assert error["message"] == "Internal server error."


# --- Readiness ----------------------------------------------------------------------------


def test_ready_when_database_and_redis_answer(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {"database": "ok", "redis": "ok", "models": "lazy"}


def test_not_ready_when_the_database_is_unreachable(client):
    class DeadSession:
        def execute(self, *args, **kwargs):
            raise ConnectionError("connection refused")

    client.app.dependency_overrides[get_db] = lambda: DeadSession()

    error = assert_envelope(client.get("/ready"), 503, "not_ready")
    assert error["checks"]["database"] == "unreachable"


def test_not_ready_when_redis_is_unreachable(client):
    class DeadRedis:
        def ping(self):
            raise RedisError("connection refused")

    client.app.dependency_overrides[get_redis] = lambda: DeadRedis()

    error = assert_envelope(client.get("/ready"), 503, "not_ready")
    assert error["checks"] == {"database": "ok", "redis": "unreachable", "models": "lazy"}


def test_health_needs_nothing(client):
    client.app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, database_url=TEST_DATABASE_URL
    )
    assert client.get("/health").json() == {"status": "ok"}
