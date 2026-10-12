"""Chat persistence and history (components 6A/6B): every /chat is stored with its trace."""

import json
import uuid

import pytest
from sqlalchemy import select

from app.api.deps import get_llm
from app.db.models import ChatQuery
from app.db.repositories import users as users_repo
from app.rag.llm import LLMError
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration

COOKIES = b"# Cookies\n\nDeclare cookie parameters with Cookie, the same way as Query parameters.\n"
ANSWER = json.dumps({"answer": "Use Cookie [1].", "citations": [1], "abstain": False})
STAGES = {"embed", "retrieve_dense", "retrieve_keyword", "fuse", "rerank", "generate", "total"}


class UnreachableLLM:
    model = "unreachable"

    def generate(self, *args, **kwargs):
        raise LLMError("connection refused")


def ask(client, question="How do I declare a cookie?", **body):
    return client.post("/chat", json={"question": question, **body})


@pytest.fixture
def answering(client):
    client.app.dependency_overrides[get_llm] = lambda: FakeLLM(ANSWER)


def stored(db_session) -> list[ChatQuery]:
    db_session.expire_all()
    return list(db_session.scalars(select(ChatQuery).order_by(ChatQuery.created_at)))


def test_answer_is_stored_with_its_full_trace(client, ingest, answering, db_session):
    ingest("cookies.md", COOKIES)

    response = ask(client)

    body = response.json()
    (row,) = stored(db_session)
    assert body["id"] == str(row.id)
    assert body["request_id"] == response.headers["X-Request-ID"] == row.request_id
    assert row.question == "How do I declare a cookie?"
    assert (row.answer, row.abstained, row.error_code) == ("Use Cookie [1].", False, None)
    assert row.citations[0]["marker"] == 1 and row.citations[0]["quote"]
    assert row.retrieved and "text" not in row.retrieved[0]
    assert {"rank", "score", "debug", "chunk_id"} <= set(row.retrieved[0])
    assert row.retrieval_config["mode"] == "hybrid"
    assert row.prompt_version == "answer_v2" and row.llm_model == "fake-llm"
    assert row.prompt_tokens > 0 and row.completion_tokens > 0
    assert set(row.timings_ms) >= STAGES


def test_timings_show_generation_separately_from_the_total(client, ingest, answering):
    ingest("cookies.md", COOKIES)

    timings = ask(client).json()["timings_ms"]

    assert set(timings) >= STAGES
    assert timings["generate"] <= timings["total"]


def test_cost_is_null_for_a_model_without_a_price(client, ingest, answering):
    ingest("cookies.md", COOKIES)

    usage = ask(client).json()["usage"]

    assert usage["cost_usd"] is None  # "fake-llm" is not in configs/pricing.toml


def test_abstention_is_stored(client, db_session):
    response = ask(client)  # no documents: abstains without calling the model

    (row,) = stored(db_session)
    assert response.json()["abstained"] is True
    assert (row.abstained, row.abstain_reason, row.error_code) == (True, "no_sources", None)


def test_failure_is_stored_with_its_error_code_and_the_timings_so_far(client, ingest, db_session):
    ingest("cookies.md", COOKIES)
    client.app.dependency_overrides[get_llm] = lambda: UnreachableLLM()

    response = ask(client)

    (row,) = stored(db_session)
    assert response.status_code == 503
    assert (row.error_code, row.answer, row.llm_model) == ("llm_unavailable", None, "unreachable")
    assert {"embed", "rerank", "total"} <= set(row.timings_ms)  # it got that far
    assert "generate" in row.timings_ms


def test_unknown_document_is_stored_as_a_failure(client, db_session):
    response = ask(client, document_ids=[str(uuid.uuid4())])

    (row,) = stored(db_session)
    assert response.status_code == 404
    assert row.error_code == "document_not_found"


def test_history_lists_newest_first_with_pagination(client, ingest, answering):
    ingest("cookies.md", COOKIES)
    for question in ("first question?", "second question?", "third question?"):
        ask(client, question)

    page = client.get("/chats", params={"limit": 2}).json()

    assert page["total"] == 3
    assert [item["question"] for item in page["items"]] == ["third question?", "second question?"]


def test_one_chat_shows_the_trace(client, ingest, answering):
    ingest("cookies.md", COOKIES)
    chat_id = ask(client).json()["id"]

    trace = client.get(f"/chats/{chat_id}").json()

    assert trace["answer"] == "Use Cookie [1]."
    assert trace["citations"][0]["marker"] == 1
    assert trace["retrieved"][0]["rank"] == 1
    assert "text" not in trace["retrieved"][0]
    assert trace["timings_ms"]["total"] > 0


def test_other_users_chats_are_invisible(client, ingest, answering, db_session, other_user):
    ingest("cookies.md", COOKIES)
    chat_id = ask(client).json()["id"]
    stranger = {"Authorization": f"Bearer {users_repo.create_api_key(db_session, other_user.id)}"}

    assert client.get("/chats", headers=stranger).json()["total"] == 0
    assert client.get(f"/chats/{chat_id}", headers=stranger).status_code == 404
