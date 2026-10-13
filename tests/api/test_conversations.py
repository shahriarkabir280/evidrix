"""Follow-up questions in a conversation (component 8D, conversation memory)."""

import json
import uuid

import pytest
from sqlalchemy import select

from app.api.deps import get_llm
from app.db.models import ChatQuery
from app.db.repositories import users as users_repo
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration

COOKIES = b"# Cookies\n\nDeclare cookie parameters with Cookie, the same way as Query parameters.\n"
HEADERS = (
    b"# Headers\n\nDeclare header parameters with Header, the same way as Cookie parameters.\n"
)


def answer(text: str) -> str:
    return json.dumps({"answer": text, "citations": [1], "abstain": False})


def rewrite(query: str) -> str:
    return json.dumps({"query": query})


def use_llm(client, *replies) -> FakeLLM:
    llm = FakeLLM(list(replies))
    client.app.dependency_overrides[get_llm] = lambda: llm
    return llm


def ask(client, question, headers=None, **body):
    return client.post("/chat", json={"question": question, **body}, headers=headers)


@pytest.fixture
def documents(ingest):
    ingest("cookies.md", COOKIES)
    ingest("headers.md", HEADERS)


def test_first_question_starts_a_conversation_without_a_rewrite(client, documents):
    llm = use_llm(client, answer("Use Cookie [1]."))

    body = ask(client, "How do I declare cookie parameters?").json()

    assert uuid.UUID(body["conversation_id"])
    assert body["search_query"] is None
    assert len(llm.calls) == 1  # the answer only


def test_follow_up_is_rewritten_then_searched_and_answered(client, documents, db_session):
    use_llm(client, answer("Use Cookie [1]."))
    first = ask(client, "How do I declare cookie parameters?").json()
    llm = use_llm(client, rewrite("How do I declare header parameters?"), answer("Use Header [1]."))

    response = ask(client, "And for headers?", conversation_id=first["conversation_id"])

    body = response.json()
    assert body["conversation_id"] == first["conversation_id"]
    assert body["search_query"] == "How do I declare header parameters?"
    assert "How do I declare cookie parameters?" in llm.calls[0]["user"]  # the rewrite
    assert "Question: How do I declare header parameters?" in llm.calls[1]["user"]  # the answer
    assert body["retrieved"][0]["document_title"] == "Headers"
    assert body["usage"]["llm_calls"] == 2
    assert "rewrite" in body["timings_ms"]
    stored = db_session.scalar(select(ChatQuery).where(ChatQuery.id == uuid.UUID(body["id"])))
    assert (stored.question, stored.search_query) == (
        "And for headers?",
        "How do I declare header parameters?",
    )


def test_a_conversation_lists_its_turns(client, documents):
    use_llm(client, answer("Use Cookie [1]."))
    first = ask(client, "How do I declare cookie parameters?").json()
    use_llm(client, rewrite("How do I declare header parameters?"), answer("Use Header [1]."))
    ask(client, "And for headers?", conversation_id=first["conversation_id"])
    use_llm(client, answer("Use Cookie [1]."))
    ask(client, "A separate question about cookies?")

    page = client.get("/chats", params={"conversation_id": first["conversation_id"]}).json()

    assert page["total"] == 2
    assert [item["question"] for item in page["items"]] == [
        "And for headers?",
        "How do I declare cookie parameters?",
    ]
    assert {item["conversation_id"] for item in page["items"]} == {first["conversation_id"]}


def test_unknown_conversation_is_not_found(client, documents):
    response = ask(client, "And for headers?", conversation_id=str(uuid.uuid4()))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


def test_another_users_conversation_is_not_found(client, documents, db_session, other_user):
    use_llm(client, answer("Use Cookie [1]."))
    first = ask(client, "How do I declare cookie parameters?").json()
    stranger = {"Authorization": f"Bearer {users_repo.create_api_key(db_session, other_user.id)}"}

    response = ask(
        client, "And for headers?", conversation_id=first["conversation_id"], headers=stranger
    )

    assert response.status_code == 404


def test_a_failed_rewrite_falls_back_to_the_question(client, documents):
    use_llm(client, answer("Use Cookie [1]."))
    first = ask(client, "How do I declare cookie parameters?").json()
    use_llm(client, "not json", answer("Use Header [1]."))

    body = ask(client, "header parameters?", conversation_id=first["conversation_id"]).json()

    assert body["search_query"] is None
    assert body["answer"] == "Use Header [1]."
