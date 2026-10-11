import pytest
from sqlalchemy import text

from app.api.deps import DEV_USER_ID
from app.db.repositories import index_versions as index_versions_repo
from app.rag.retrieval import keyword_search
from app.services import ingestion

pytestmark = pytest.mark.integration

DOCUMENTS = {
    "oauth.md": (
        "# Simple OAuth2\n\nUse OAuth2PasswordRequestForm to read the username and password "
        "sent as form data.\n"
    ),
    "cookies.md": "# Cookies\n\nDeclare cookie parameters with Cookie, like Query parameters.\n",
    "errors.md": (
        "# Handling Errors\n\nRaise HTTPException with status_code=404 when an item is missing. "
        "An HTTPException can also carry headers.\n"
    ),
    "tasks.md": "# Background Tasks\n\nRun work after returning a response with BackgroundTasks.\n",
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


def search(session_factory, question, *, owner_id=DEV_USER_ID, **kwargs):
    with session_factory() as session:
        version = index_versions_repo.get_active(session).id
        return keyword_search(
            session, owner_id=owner_id, index_version_id=version, question=question, **kwargs
        )


def test_exact_identifier_is_found(session_factory, corpus):
    hits = search(session_factory, "What is OAuth2PasswordRequestForm?")

    assert hits[0].document_id == corpus["oauth.md"]


def test_different_word_forms_match_through_stemming(session_factory, corpus):
    hits = search(session_factory, "declaring cookies")

    assert hits and hits[0].document_id == corpus["cookies.md"]


def test_one_unknown_word_does_not_wipe_out_the_results(session_factory, corpus):
    hits = search(session_factory, "How do I declare cookie parameters in FastAPI quickly?")

    assert hits[0].document_id == corpus["cookies.md"]


def test_more_matching_words_rank_higher(session_factory, corpus):
    hits = search(session_factory, "raise HTTPException status_code 404")

    assert hits[0].document_id == corpus["errors.md"]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))


def test_question_made_only_of_stop_words_returns_nothing(session_factory, corpus):
    assert search(session_factory, "how do I do it") == []


@pytest.mark.parametrize("question", ["cookie & | ! ( ) :* <->", "'; DROP TABLE chunks; --", ""])
def test_any_user_text_is_safe_to_search_with(session_factory, corpus, question):
    search(session_factory, question)  # must not raise

    with session_factory() as session:
        assert session.scalar(text("SELECT count(*) FROM chunks")) > 0


def test_document_filter_restricts_results(session_factory, corpus):
    hits = search(session_factory, "cookie parameters", document_ids=[corpus["errors.md"]])

    assert {h.document_id for h in hits} <= {corpus["errors.md"]}


def test_other_users_chunks_are_never_returned(
    session_factory, blob_store, embedder, corpus, other_user
):
    with session_factory() as session:
        ingestion.ingest_upload(
            session,
            blob_store,
            embedder,
            owner_id=other_user.id,
            filename="private.md",
            data=b"# Private\n\nThe OAuth2PasswordRequestForm notes of another user.\n",
        )

    hits = search(session_factory, "OAuth2PasswordRequestForm", k=10)

    assert {h.document_id for h in hits} == {corpus["oauth.md"]}


def test_documents_that_are_not_completed_are_excluded(session_factory, corpus):
    with session_factory() as session:
        session.execute(
            text("UPDATE documents SET status = 'failed' WHERE id = :id"),
            {"id": corpus["oauth.md"]},
        )
        session.commit()

    assert search(session_factory, "OAuth2PasswordRequestForm") == []


def test_section_heading_words_outweigh_body_words(session_factory, blob_store, embedder):
    with session_factory() as session:
        for filename, body in {
            "heading.md": "# Middleware\n\nA function that runs for every request.\n",
            "body.md": "# Requests\n\nYou could also use middleware for this one request.\n",
        }.items():
            ingestion.ingest_upload(
                session,
                blob_store,
                embedder,
                owner_id=DEV_USER_ID,
                filename=filename,
                data=body.encode(),
            )

    hits = search(session_factory, "middleware")

    assert hits[0].section_path == "Middleware"
