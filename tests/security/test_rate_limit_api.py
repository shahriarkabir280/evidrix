"""Rate limits on the API (component 5E): 429 with Retry-After, per user and per route."""

import pytest

from app.db.repositories import users as users_repo
from tests.conftest import TEST_MAX_UPLOAD_BYTES

pytestmark = pytest.mark.integration

COOKIES = b"# Cookies\n\nDeclare cookie parameters with the Cookie class.\n"


def ask(client, **kwargs):
    return client.post("/chat", json={"question": "How do I declare cookies?"}, **kwargs)


def test_21st_chat_in_a_minute_is_refused_with_retry_after(client, ingest):
    ingest("cookies.md", COOKIES)
    responses = [ask(client) for _ in range(21)]

    assert [r.status_code for r in responses[:20]] == [200] * 20
    refused = responses[20]
    assert refused.status_code == 429
    assert refused.json()["detail"]["code"] == "rate_limited"
    assert 1 <= int(refused.headers["Retry-After"]) <= 60
    assert refused.headers["X-RateLimit-Remaining"] == "0"


def test_successful_responses_report_the_remaining_budget(client):
    first = client.post("/retrieve", json={"question": "cookies"})
    second = client.post("/retrieve", json={"question": "cookies"})

    assert first.headers["X-RateLimit-Limit"] == "60"
    assert (first.headers["X-RateLimit-Remaining"], second.headers["X-RateLimit-Remaining"]) == (
        "59",
        "58",
    )


def test_one_users_limit_does_not_affect_another(client, db_session, other_user):
    for _ in range(20):
        ask(client)
    other = {"Authorization": f"Bearer {users_repo.create_api_key(db_session, other_user.id)}"}

    assert ask(client).status_code == 429
    assert ask(client, headers=other).status_code == 200


def test_routes_have_separate_budgets(client):
    for _ in range(21):
        ask(client)

    assert client.post("/retrieve", json={"question": "cookies"}).status_code == 200


def test_eleventh_upload_in_a_minute_is_refused(client):
    statuses = [
        client.post(
            "/documents", files={"file": (f"n{i}.md", f"# Note {i}\n\nText.".encode())}
        ).status_code
        for i in range(11)
    ]

    assert statuses == [202] * 10 + [429]


def test_rejected_and_unauthenticated_requests_do_not_spend_the_budget(client):
    for _ in range(20):
        client.post("/chat", json={"question": "x"}, headers={"Authorization": "Bearer nope"})

    assert ask(client).headers["X-RateLimit-Remaining"] == "19"


def test_reading_documents_is_not_limited(client):
    statuses = {client.get("/documents").status_code for _ in range(30)}

    assert statuses == {200}


def test_oversized_upload_still_counts(client):
    # The limit is checked before the body is read, so large rejected uploads cost budget too:
    # otherwise they'd be a free way to make the server read megabytes.
    client.post("/documents", files={"file": ("big.txt", b"a" * (TEST_MAX_UPLOAD_BYTES + 1))})

    response = client.post("/documents", files={"file": ("ok.md", COOKIES)})

    assert response.status_code == 202
    assert response.headers["X-RateLimit-Remaining"] == "8"  # 10 - the rejected one - this one
