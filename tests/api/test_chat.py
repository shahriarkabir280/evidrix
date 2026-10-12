import json
import uuid

import pytest
from sqlalchemy import text

from app.api.deps import get_llm
from app.rag.generation import ABSTAIN_MESSAGE
from app.rag.llm import LLMError
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration

COOKIES = b"# Cookies\n\nDeclare cookie parameters with Cookie, the same way as Query parameters.\n"
CORS = b"# CORS\n\nAllow cross-origin browser requests by adding CORSMiddleware to the app.\n"
SECRET = b"# Internal\n\nThe internal deployment password is swordfish-42 and must not leak.\n"


def reply(answer: str, citations: list[int], abstain: bool = False) -> str:
    return json.dumps({"answer": answer, "citations": citations, "abstain": abstain})


def use_llm(client, llm) -> object:
    client.app.dependency_overrides[get_llm] = lambda: llm
    return llm


def ask(client, question: str, **body):
    return client.post("/chat", json={"question": question, **body})


# --- Answers -----------------------------------------------------------------------------


def test_answer_comes_back_with_citations_and_the_evidence(client, ingest):
    ingest("cookies.md", COOKIES)
    ingest("cors.md", CORS)
    llm = use_llm(client, FakeLLM(reply("Declare them with Cookie [1].", [1])))

    response = ask(client, "How do I declare cookie parameters?")

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["answer"] == "Declare them with Cookie [1]."
    citation = body["citations"][0]
    assert citation["marker"] == 1
    assert citation["document_title"] == "Cookies"
    assert citation["section_path"] == "Cookies"
    assert "Declare cookie parameters" in citation["quote"]
    assert body["retrieved"][0]["chunk_id"] == citation["chunk_id"]
    assert [r["rank"] for r in body["retrieved"]] == list(range(1, len(body["retrieved"]) + 1))
    # The default preset is hybrid + rerank, so every stage reports its own time.
    assert set(body["timings_ms"]) == {
        "embed",
        "retrieve_dense",
        "retrieve_keyword",
        "fuse",
        "rerank",
        "generate",
        "total",
    }
    assert body["config"]["retrieval"]["mode"] == "hybrid"
    assert body["config"]["reranker_model"] == "fake-word-overlap"
    assert body["usage"]["llm_calls"] == 1
    assert body["config"]["prompt_version"] == "answer_v2"
    assert "Declare cookie parameters" in llm.calls[0]["user"]


def test_answer_can_cite_several_passages(client, ingest):
    ingest("cookies.md", COOKIES)
    ingest("cors.md", CORS)
    use_llm(client, FakeLLM(reply("Cookies use Cookie [1]; CORS uses CORSMiddleware [2].", [1, 2])))

    body = ask(client, "cookie parameters and cors middleware").json()

    cited_documents = {c["document_id"] for c in body["citations"]}
    assert len(cited_documents) == 2


def test_model_abstention_is_returned_as_is(client, ingest):
    ingest("cookies.md", COOKIES)
    use_llm(client, FakeLLM(reply("", [], abstain=True)))

    body = ask(client, "What rate limit does FastAPI apply?").json()

    assert body["abstained"] is True
    assert body["abstain_reason"] == "model"
    assert body["answer"] == ABSTAIN_MESSAGE
    assert body["citations"] == []


def test_without_any_documents_the_model_is_not_called(client, llm):
    body = ask(client, "anything").json()

    assert body["abstain_reason"] == "no_sources"
    assert llm.calls == []


# --- Document selection ------------------------------------------------------------------


def test_only_selected_documents_reach_the_model(client, ingest):
    cookies = ingest("cookies.md", COOKIES)
    ingest("internal.md", SECRET)
    llm = use_llm(client, FakeLLM(reply("", [], abstain=True)))

    body = ask(client, "What is the internal deployment password?", document_ids=[cookies]).json()

    assert {r["document_id"] for r in body["retrieved"]} == {cookies}
    assert "swordfish" not in llm.calls[0]["user"]


def test_duplicate_document_ids_are_accepted(client, ingest):
    cookies = ingest("cookies.md", COOKIES)

    assert ask(client, "cookies", document_ids=[cookies, cookies]).status_code == 200


def test_unknown_document_is_not_found(client):
    response = ask(client, "cookies", document_ids=[str(uuid.uuid4())])

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_another_users_document_is_not_found(client, ingest, db_session, other_user):
    document_id = ingest("cookies.md", COOKIES)
    db_session.execute(
        text("UPDATE documents SET owner_id = :owner WHERE id = :id"),
        {"owner": other_user.id, "id": document_id},
    )
    db_session.commit()

    assert ask(client, "cookies", document_ids=[document_id]).status_code == 404


def test_document_still_processing_is_a_conflict(client, ingest, db_session):
    document_id = ingest("cookies.md", COOKIES)
    db_session.execute(
        text("UPDATE documents SET status = 'processing' WHERE id = :id"), {"id": document_id}
    )
    db_session.commit()

    response = ask(client, "cookies", document_ids=[document_id])

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "document_not_ready"


# --- Invalid requests --------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "ok", "document_ids": []},
        {"question": "ok", "document_ids": ["not-a-uuid"]},
        {},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert client.post("/chat", json=body).status_code == 422


# --- Model failures ----------------------------------------------------------------------


class UnreachableLLM:
    model = "unreachable"

    def generate(self, *args, **kwargs):
        raise LLMError("connection refused")


def test_unreachable_model_is_service_unavailable(client, ingest):
    ingest("cookies.md", COOKIES)
    use_llm(client, UnreachableLLM())

    response = ask(client, "cookies")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


def test_unusable_model_output_is_bad_gateway(client, ingest):
    ingest("cookies.md", COOKIES)
    use_llm(client, FakeLLM(["not json", "still not json"]))

    response = ask(client, "cookies")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_bad_output"


def test_preset_threshold_abstains_without_calling_the_model(client, ingest, monkeypatch):
    from app.api import deps
    from app.rag.retrieval import RetrievalConfig

    ingest("cookies.md", COOKIES)
    llm = use_llm(client, FakeLLM(reply("never used", [1])))
    strict = RetrievalConfig(mode="dense", rerank=False, min_score=2.0)  # cosine never reaches 2
    monkeypatch.setattr(deps, "load_preset", lambda name: strict)

    body = ask(client, "How do I declare cookie parameters?").json()

    assert body["abstain_reason"] == "low_retrieval_score"
    assert body["config"]["retrieval"]["min_score"] == 2.0
    assert llm.calls == []
