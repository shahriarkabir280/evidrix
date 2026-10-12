"""Prompt injection against the real model (component 5F). Needs Ollama: `make test-live`.

Retrieval uses the fake embedder and reranker, but production chunk sizes: with the tests' tiny
chunks an attack lands in a source of its own, apart from the text that answers the question,
and the model ignores it for that reason alone (this test then passed with the v1 prompt, which
follows the citation-rule attack in production).
"""

import json

import pytest
from sqlalchemy import text

from app.core.config import get_settings
from app.rag.llm import get_llm
from app.rag.retrieval import load_preset
from evals.injection import run_suite

pytestmark = [pytest.mark.integration, pytest.mark.live]

PRODUCTION_CHUNKING = {
    "chunk_size_tokens": 400,
    "max_tokens": 480,
    "overlap_tokens": 64,
    "min_chunk_tokens": 100,
    "section_level": 2,
}


@pytest.fixture
def production_chunking(db_session, embedder):
    db_session.execute(text("UPDATE index_versions SET is_active = false"))
    db_session.execute(
        text(
            "INSERT INTO index_versions "
            "(name, embedding_model, embedding_dim, chunking, contextual_header, is_active) "
            "VALUES ('test-live-production-chunking', :model, :dim, CAST(:chunking AS jsonb), "
            "true, true)"
        ),
        {
            "model": embedder.model_name,
            "dim": embedder.dim,
            "chunking": json.dumps(PRODUCTION_CHUNKING),
        },
    )
    db_session.commit()


def test_injected_instructions_are_not_followed(
    production_chunking, db_session, blob_store, embedder, reranker
):
    settings = get_settings()

    rows = run_suite(
        db_session,
        blob_store,
        embedder,
        reranker,
        get_llm(),
        retrieval=load_preset(settings.retrieval_preset),
        max_tokens=settings.llm_max_output_tokens,
        prompt_version=settings.answer_prompt_version,
    )

    assert all(row["injection_in_prompt"] and row["model_called"] for row in rows)
    followed = [row["id"] for row in rows if row["kind"] == "instruction" and row["hijacked"]]
    assert followed == []
