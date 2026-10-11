import uuid

import pytest

from app.api.deps import get_llm
from app.rag.llm import PromptTooLongError

pytestmark = pytest.mark.integration

COOKIES = b"# Cookies\n\nDeclare cookie parameters with Cookie, the same way as Query parameters.\n"
CORS = b"# CORS\n\nAllow cross-origin browser requests by adding CORSMiddleware to the app.\n"


def upload(client, filename, data):
    return client.post("/documents", files={"file": (filename, data)}).json()["id"]


def test_retrieve_returns_ranked_chunks_with_debug_and_never_calls_the_model(client, llm):
    upload(client, "cookies.md", COOKIES)
    upload(client, "cors.md", CORS)

    response = client.post("/retrieve", json={"question": "cookie parameters"})

    assert response.status_code == 200
    body = response.json()
    top = body["hits"][0]
    assert top["document_title"] == "Cookies"
    assert {"dense_rank", "rrf_score", "rerank_score", "pre_rerank_rank"} <= set(top["debug"])
    assert body["config"]["retrieval"]["rerank"] is True
    assert "rerank" in body["timings_ms"]
    assert llm.calls == []


def test_request_can_switch_to_keyword_only(client):
    upload(client, "cookies.md", COOKIES)

    body = client.post(
        "/retrieve",
        json={"question": "cookie", "retrieval": {"mode": "keyword", "rerank": False, "top_k": 1}},
    ).json()

    assert body["config"]["retrieval"]["mode"] == "keyword"
    assert body["config"]["reranker_model"] is None
    assert set(body["hits"][0]["debug"]) == set()  # a single retriever adds no fusion signals
    assert len(body["hits"]) == 1


@pytest.mark.parametrize(
    "retrieval",
    [
        {"top_k": 50},
        {"top_k": 10, "candidate_k": 5},
        {"mode": "semantic"},
        {"temperature": 1},
    ],
)
def test_invalid_overrides_are_rejected(client, retrieval):
    response = client.post("/retrieve", json={"question": "cookie", "retrieval": retrieval})

    assert response.status_code == 422


def test_unknown_document_is_not_found(client):
    response = client.post(
        "/retrieve", json={"question": "cookie", "document_ids": [str(uuid.uuid4())]}
    )

    assert response.status_code == 404


def test_chat_uses_the_overridden_retrieval(client):
    upload(client, "cookies.md", COOKIES)

    body = client.post(
        "/chat", json={"question": "cookie", "retrieval": {"mode": "dense", "rerank": False}}
    ).json()

    assert body["config"]["retrieval"]["mode"] == "dense"
    assert set(body["timings_ms"]) == {"embed", "retrieve_dense", "generate", "total"}


class TooLongLLM:
    model = "tiny-context"

    def generate(self, *args, **kwargs):
        raise PromptTooLongError("prompt does not fit")


def test_prompt_too_long_for_the_model_is_a_client_error(client):
    upload(client, "cookies.md", COOKIES)
    client.app.dependency_overrides[get_llm] = lambda: TooLongLLM()

    response = client.post("/chat", json={"question": "cookie", "retrieval": {"top_k": 20}})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "prompt_too_long"
