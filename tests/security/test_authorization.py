"""Authorization audit (component 5D): no path lets user B read user A's content.

User A is the dev user (the client's default key); user B gets a key of their own. Access
control lives in SQL, so the last tests call the repositories directly, as a route that forgot
its check would.
"""

import inspect
import uuid

import numpy as np
import pytest

from app.db.repositories import chats as chats_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import documents as documents_repo
from app.db.repositories import index_versions as index_versions_repo
from app.db.repositories import jobs as jobs_repo
from app.db.repositories import users as users_repo

pytestmark = pytest.mark.integration

CANARY = "zebracanary4471"
A_DOC = f"# Private notes\n\nThe launch code word is {CANARY}, known only to user A.\n".encode()
B_DOC = b"# B notes\n\nUser B writes about launch plans and code words of their own.\n"


@pytest.fixture
def b_auth(db_session, other_user) -> dict[str, str]:
    return {"Authorization": f"Bearer {users_repo.create_api_key(db_session, other_user.id)}"}


@pytest.fixture
def a_upload(client, run_worker) -> dict:
    """A's document, processed; returns the upload response (id, job_id, ...)."""
    body = client.post("/documents", files={"file": ("a.md", A_DOC)}).json()
    run_worker()
    return body


def b_ingest(client, run_worker, b_auth) -> str:
    response = client.post("/documents", files={"file": ("b.md", B_DOC)}, headers=b_auth)
    assert response.status_code == 202
    run_worker()
    return response.json()["id"]


# --- Every route that takes an id answers B's request for A's resource with 404 -----------


@pytest.mark.parametrize(
    "request_for",
    [
        lambda a: ("GET", f"/documents/{a['id']}", None),
        lambda a: ("GET", f"/jobs/{a['job_id']}", None),
        lambda a: ("POST", "/chat", {"question": "code word?", "document_ids": [a["id"]]}),
        lambda a: ("POST", "/retrieve", {"question": "code word?", "document_ids": [a["id"]]}),
    ],
    ids=["document", "job", "chat", "retrieve"],
)
def test_other_users_resource_looks_exactly_like_a_missing_one(
    client, a_upload, b_auth, request_for
):
    method, path, body = request_for(a_upload)
    missing = {key: str(uuid.uuid4()) for key in ("id", "job_id")}
    method_m, path_m, body_m = request_for(missing)

    foreign = client.request(method, path, json=body, headers=b_auth)
    absent = client.request(method_m, path_m, json=body_m, headers=b_auth)

    assert foreign.status_code == absent.status_code == 404
    assert foreign.json()["error"]["code"] == absent.json()["error"]["code"]


def test_owner_still_sees_their_own_resources(client, a_upload):
    assert client.get(f"/documents/{a_upload['id']}").status_code == 200
    assert client.get(f"/jobs/{a_upload['job_id']}").status_code == 200


def test_listing_shows_only_your_documents(client, a_upload, run_worker, b_auth):
    b_id = b_ingest(client, run_worker, b_auth)

    listing = client.get("/documents", headers=b_auth).json()

    assert [item["id"] for item in listing["items"]] == [b_id]


# --- Searching without a document filter never reaches another user's chunks --------------


def test_unfiltered_retrieve_returns_only_your_chunks(client, a_upload, run_worker, b_auth):
    b_id = b_ingest(client, run_worker, b_auth)

    response = client.post(
        "/retrieve", json={"question": f"launch code word {CANARY}"}, headers=b_auth
    )

    hits = response.json()["hits"]
    assert hits, "B's own document should match"
    assert {hit["document_id"] for hit in hits} == {b_id}
    assert all(CANARY not in hit["text"] for hit in hits)


def test_other_users_text_never_reaches_the_prompt(client, a_upload, run_worker, b_auth, llm):
    b_ingest(client, run_worker, b_auth)

    client.post("/chat", json={"question": f"What is the code word {CANARY}?"}, headers=b_auth)

    assert llm.calls, "B has a document, so the model is called"
    sources = llm.calls[0]["user"].split("Question:")[0]
    assert CANARY not in sources


# --- Deduplication is per user ------------------------------------------------------------


def test_same_file_from_another_user_is_a_new_document(client, a_upload, b_auth):
    response = client.post("/documents", files={"file": ("copy.md", A_DOC)}, headers=b_auth)

    # A global dedup would answer "duplicate" and reveal that someone already has this file.
    assert response.status_code == 202
    assert response.json()["duplicate"] is False
    assert response.json()["id"] != a_upload["id"]


# --- The filter is in SQL: a route that forgot its check still leaks nothing --------------


def test_search_with_another_users_document_id_returns_nothing(
    client, a_upload, db_session, other_user
):
    index_version_id = index_versions_repo.get_active(db_session).id
    common = {
        "owner_id": other_user.id,
        "index_version_id": index_version_id,
        "k": 10,
        "document_ids": [uuid.UUID(a_upload["id"])],
    }

    dense = chunks_repo.search_dense(db_session, query_vector=np.ones(384), **common)
    keyword = chunks_repo.search_keyword(db_session, query_text=CANARY, **common)

    assert dense == keyword == []


# --- Policy: every repository function a request can reach is scoped by owner -------------

# Functions that act for the system (worker, reaper, query building), never for a request.
SYSTEM_ONLY = {"jobs.claim", "jobs.stale", "chunks.or_tsquery"}


@pytest.mark.parametrize(
    "module", [documents_repo, chunks_repo, jobs_repo, chats_repo], ids=lambda m: m.__name__
)
def test_repository_reads_of_user_data_require_owner_id(module):
    short = module.__name__.rsplit(".", 1)[1]
    unscoped = [
        f"{short}.{name}"
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module.__name__
        and not name.startswith("_")
        and f"{short}.{name}" not in SYSTEM_ONLY
        and "owner_id" not in inspect.signature(function).parameters
    ]

    assert unscoped == [], "add owner_id, or list the function in SYSTEM_ONLY with a reason"
