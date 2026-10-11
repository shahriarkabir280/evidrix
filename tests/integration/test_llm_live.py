"""Calls the real model behind LLM_BASE_URL (local Ollama). Run with `make test-live`."""

import json
import uuid

import pytest

from app.db.repositories.chunks import ChunkHit
from app.rag.generation import ANSWER_SCHEMA, build_answer_prompt
from app.rag.llm import get_llm

pytestmark = pytest.mark.live

SOURCES = [
    "Use OAuth2PasswordRequestForm as a dependency to receive the username and password "
    "that the client sends as form data in the OAuth2 password flow.",
    "Background tasks run after the response has been sent to the client.",
]


def hits() -> list[ChunkHit]:
    return [
        ChunkHit(uuid.uuid4(), uuid.uuid4(), "Simple OAuth2", 0, text, None, None, None, 0.8, rank)
        for rank, text in enumerate(SOURCES, start=1)
    ]


def ask(question: str) -> dict:
    system, user = build_answer_prompt(question, hits())
    result = get_llm().generate(system, user, max_tokens=300, json_schema=ANSWER_SCHEMA)
    assert result.finish_reason == "stop"
    return json.loads(result.text)


def test_answerable_question_is_answered_with_a_citation():
    reply = ask("What does OAuth2PasswordRequestForm receive?")

    assert reply["abstain"] is False
    assert 1 in reply["citations"]
    assert "password" in reply["answer"].lower()


def test_unanswerable_question_abstains():
    reply = ask("Which database does FastAPI use by default to store users?")

    assert reply["abstain"] is True
    assert reply["citations"] == []
