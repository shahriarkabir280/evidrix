import pytest

from app.api.deps import DEV_USER_ID
from app.db.repositories import index_versions as index_versions_repo
from app.rag.retrieval import hybrid_search
from app.services import ingestion

pytestmark = pytest.mark.integration

DOCUMENTS = {
    "oauth.md": "# Simple OAuth2\n\nUse OAuth2PasswordRequestForm to read the username.\n",
    "cookies.md": "# Cookies\n\nDeclare cookie parameters with Cookie, like Query parameters.\n",
    "tasks.md": "# Background Tasks\n\nRun work after returning a response.\n",
}


@pytest.fixture
def corpus(session_factory, blob_store, embedder):
    ids = {}
    with session_factory() as session:
        for filename, body in DOCUMENTS.items():
            ids[filename] = ingestion.ingest_upload(
                session,
                blob_store,
                embedder,
                owner_id=DEV_USER_ID,
                filename=filename,
                data=body.encode(),
            ).document.id
    return ids


def search(session_factory, embedder, question, *, owner_id=DEV_USER_ID, **kwargs):
    with session_factory() as session:
        version = index_versions_repo.get_active(session).id
        return hybrid_search(
            session,
            embedder,
            owner_id=owner_id,
            index_version_id=version,
            question=question,
            **kwargs,
        )


def test_hybrid_results_carry_both_retrievers_signals(session_factory, embedder, corpus):
    hits = search(session_factory, embedder, "OAuth2PasswordRequestForm username")

    top = hits[0]
    assert top.document_id == corpus["oauth.md"]
    assert {"dense_rank", "keyword_rank", "rrf_score"} <= set(top.debug)
    assert top.score == pytest.approx(top.debug["rrf_score"])


def test_k_limits_the_fused_list(session_factory, embedder, corpus):
    assert len(search(session_factory, embedder, "cookie response username", k=2)) == 2


def test_both_legs_respect_the_owner(session_factory, blob_store, embedder, corpus, other_user):
    with session_factory() as session:
        ingestion.ingest_upload(
            session,
            blob_store,
            embedder,
            owner_id=other_user.id,
            filename="private.md",
            data=b"# Private\n\nOAuth2PasswordRequestForm username notes of another user.\n",
        )

    hits = search(session_factory, embedder, "OAuth2PasswordRequestForm username", k=10)

    assert {h.document_id for h in hits} <= set(corpus.values())


def test_both_legs_respect_the_document_filter(session_factory, embedder, corpus):
    hits = search(
        session_factory, embedder, "OAuth2PasswordRequestForm", document_ids=[corpus["tasks.md"]]
    )

    assert {h.document_id for h in hits} <= {corpus["tasks.md"]}
