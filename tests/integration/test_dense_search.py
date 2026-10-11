import pytest
from sqlalchemy import text

from app.api.deps import DEV_USER_ID
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import index_versions as index_versions_repo
from app.rag.retrieval import dense_search
from app.services import ingestion

pytestmark = pytest.mark.integration

DOCUMENTS = {
    "cookies.md": "# Cookies\n\nDeclare cookie parameters with Cookie, like Query parameters.\n",
    "cors.md": "# CORS\n\nAllow cross-origin browser requests with CORSMiddleware.\n",
    "sorting.md": "# Sorting\n\nSort lists with sorted() and a key function such as len.\n",
}


@pytest.fixture
def corpus(session_factory, blob_store, embedder):
    """Ingest the three small documents and return {filename: document_id}."""
    ids = {}
    with session_factory() as session:
        for filename, body in DOCUMENTS.items():
            document = ingestion.ingest_upload(
                session,
                blob_store,
                embedder,
                owner_id=DEV_USER_ID,
                filename=filename,
                data=body.encode(),
            ).document
            ids[filename] = document.id
    return ids


def search(session_factory, embedder, question, *, owner_id=DEV_USER_ID, **kwargs):
    with session_factory() as session:
        index_version_id = kwargs.pop(
            "index_version_id", index_versions_repo.get_active(session).id
        )
        return dense_search(
            session,
            embedder,
            owner_id=owner_id,
            index_version_id=index_version_id,
            question=question,
            **kwargs,
        )


def test_most_similar_chunk_ranks_first(session_factory, embedder, corpus):
    hits = search(session_factory, embedder, "how do I declare cookie parameters")

    assert hits[0].document_id == corpus["cookies.md"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_k_limits_the_number_of_results(session_factory, embedder, corpus):
    assert len(search(session_factory, embedder, "cookie", k=2)) == 2


def test_document_filter_restricts_results(session_factory, embedder, corpus):
    hits = search(
        session_factory, embedder, "cookie parameters", document_ids=[corpus["sorting.md"]]
    )

    assert {h.document_id for h in hits} == {corpus["sorting.md"]}


def test_empty_document_filter_returns_nothing(session_factory, embedder, corpus):
    assert search(session_factory, embedder, "cookie", document_ids=[]) == []


def test_other_users_identical_document_is_never_returned(
    session_factory, blob_store, embedder, corpus, other_user
):
    with session_factory() as session:
        ingestion.ingest_upload(
            session,
            blob_store,
            embedder,
            owner_id=other_user.id,
            filename="cookies.md",
            data=DOCUMENTS["cookies.md"].encode(),
        )

    mine = search(session_factory, embedder, "cookie parameters", k=10)
    theirs = search(session_factory, embedder, "cookie parameters", k=10, owner_id=other_user.id)

    assert {h.document_id for h in mine} == set(corpus.values())
    assert len(theirs) == 1 and theirs[0].document_id not in corpus.values()


def test_documents_that_are_not_completed_are_excluded(session_factory, embedder, corpus):
    with session_factory() as session:
        session.execute(
            text("UPDATE documents SET status = 'processing' WHERE id = :id"),
            {"id": corpus["cookies.md"]},
        )
        session.commit()

    hits = search(session_factory, embedder, "cookie parameters", k=10)

    assert corpus["cookies.md"] not in {h.document_id for h in hits}


def test_other_index_versions_are_not_searched(session_factory, embedder, corpus):
    with session_factory() as session:
        other_version = session.scalar(
            text("SELECT id FROM index_versions WHERE name = 'bge-small_c400_o64'")
        )

    assert search(session_factory, embedder, "cookie", index_version_id=other_version) == []


def test_hnsw_and_exact_search_agree_on_a_small_index(session_factory, embedder, corpus):
    query_vector = embedder.embed_query("sort a list with a key function")
    with session_factory() as session:
        version = index_versions_repo.get_active(session).id
        common = {"owner_id": DEV_USER_ID, "index_version_id": version, "k": 3}
        approximate = chunks_repo.search_dense(session, query_vector=query_vector, **common)
    with session_factory() as session:
        exact = chunks_repo.search_dense(session, query_vector=query_vector, exact=True, **common)

    assert [h.chunk_id for h in approximate] == [h.chunk_id for h in exact]
