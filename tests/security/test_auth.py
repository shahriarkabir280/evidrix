"""API-key authentication (component 5C): who is calling, and what a bad key gets."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.cli import create_key
from app.db.models import ApiKey
from app.db.repositories import users as users_repo
from app.db.repositories.users import DEV_USER_ID

pytestmark = pytest.mark.integration

PROTECTED = [
    ("GET", "/documents"),
    ("GET", "/documents/00000000-0000-0000-0000-000000000000"),
    ("POST", "/documents"),
    ("GET", "/jobs/00000000-0000-0000-0000-000000000000"),
    ("POST", "/chat"),
    ("POST", "/retrieve"),
]


def call(client: TestClient, method: str, path: str, authorization: str | None):
    headers = {"Authorization": authorization} if authorization is not None else {}
    # Drop the client's default (valid) header, so only `headers` decides.
    client.headers.pop("Authorization", None)
    return client.request(method, path, headers=headers)


def assert_unauthorized(response):
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["detail"]["code"] == "invalid_api_key"


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_every_endpoint_except_health_needs_a_key(client, method, path):
    assert_unauthorized(call(client, method, path, authorization=None))


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer",  # no token
        "Basic ZGV2OnNlY3JldA==",  # another scheme
        "Bearer not-an-evidrix-key",  # wrong format
        "Bearer evx_" + "A" * 43,  # right format, unknown key
    ],
)
def test_bad_credentials_are_refused_the_same_way(client, authorization):
    assert_unauthorized(call(client, "GET", "/documents", authorization))


def test_health_needs_no_key(client):
    assert call(client, "GET", "/health", authorization=None).status_code == 200


def test_valid_key_identifies_its_user(client, db_session, other_user):
    key = users_repo.create_api_key(db_session, other_user.id)

    response = call(client, "GET", "/documents", f"Bearer {key}")

    assert response.status_code == 200


def test_revoked_key_is_refused(client, db_session):
    key = users_repo.create_api_key(db_session, DEV_USER_ID)
    assert call(client, "GET", "/documents", f"Bearer {key}").status_code == 200

    revoked = users_repo.revoke_api_keys(db_session, key[:12])

    assert revoked == 1
    assert_unauthorized(call(client, "GET", "/documents", f"Bearer {key}"))


def test_only_the_hash_of_a_key_is_stored(db_session):
    key = users_repo.create_api_key(db_session, DEV_USER_ID, "laptop")

    stored = db_session.scalars(select(ApiKey).where(ApiKey.name == "laptop")).one()

    assert key not in (stored.key_hash, stored.prefix)
    assert stored.prefix == key[:12]
    assert len(stored.key_hash) == 64


def test_cli_prints_a_working_key_once(client, session_factory, capsys, monkeypatch):
    monkeypatch.setattr("app.cli.get_session_factory", lambda: session_factory)

    create_key("dev@evidrix.local", "cli-test")
    key = next(word for word in capsys.readouterr().out.split() if word.startswith("evx_"))

    assert call(client, "GET", "/documents", f"Bearer {key}").status_code == 200


def test_users_do_not_see_each_others_documents(client, db_session, other_user, ingest):
    document_id = ingest("mine.md", b"# Mine\n\nA document that belongs to the dev user.")
    stranger = users_repo.create_api_key(db_session, other_user.id)

    listing = call(client, "GET", "/documents", f"Bearer {stranger}").json()
    direct = call(client, "GET", f"/documents/{document_id}", f"Bearer {stranger}")

    assert listing["total"] == 0
    assert direct.status_code == 404
