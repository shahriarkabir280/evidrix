import pytest

from app.api.deps import DEV_USER_ID
from app.db.models import Document, DocumentStatus
from app.db.repositories import index_versions as index_versions_repo
from app.rag.extract import ExtractedDocument, PageText
from app.rag.retrieval import RetrievalConfig, retrieve
from app.services import ingestion
from app.services.ingestion import build_chunks

pytestmark = pytest.mark.integration

QUESTION = "cookie parameters with Cookie"


@pytest.fixture
def corpus(session_factory, blob_store, embedder):
    with session_factory() as session:
        for n in range(8):
            body = f"# Topic {n}\n\nCookie parameters note {n}: declare them with Cookie.\n"
            ingestion.ingest_upload(
                session,
                blob_store,
                embedder,
                owner_id=DEV_USER_ID,
                filename=f"note{n}.md",
                data=body.encode(),
            )


def run(session_factory, embedder, reranker, **config):
    with session_factory() as session:
        version = index_versions_repo.get_active(session).id
        return retrieve(
            session,
            embedder,
            reranker,
            RetrievalConfig(**config),
            owner_id=DEV_USER_ID,
            index_version_id=version,
            question=QUESTION,
        )


@pytest.mark.parametrize(
    ("config", "stages"),
    [
        ({"mode": "dense"}, {"embed", "retrieve_dense"}),
        ({"mode": "keyword"}, {"retrieve_keyword"}),
        ({"mode": "hybrid"}, {"embed", "retrieve_dense", "retrieve_keyword", "fuse"}),
        (
            {"mode": "hybrid", "rerank": True},
            {"embed", "retrieve_dense", "retrieve_keyword", "fuse", "rerank"},
        ),
    ],
)
def test_each_pipeline_times_exactly_the_stages_it_runs(
    session_factory, embedder, reranker, corpus, config, stages
):
    result = run(session_factory, embedder, reranker, top_k=3, candidate_k=6, **config)

    assert set(result.timings_ms) == stages
    assert len(result.hits) == 3


def test_reranker_sees_the_whole_candidate_pool_and_returns_top_k(
    session_factory, embedder, reranker, corpus
):
    result = run(
        session_factory, embedder, reranker, mode="hybrid", rerank=True, top_k=2, candidate_k=6
    )

    assert len(reranker.calls) == 1 and len(reranker.calls[0]) == 6
    assert len(result.hits) == 2
    assert {"pre_rerank_rank", "rerank_score", "rrf_score"} <= set(result.hits[0].debug)


def test_without_rerank_the_reranker_is_never_called(session_factory, embedder, reranker, corpus):
    run(session_factory, embedder, reranker, mode="hybrid", top_k=3)

    assert reranker.calls == []


def test_reference_sections_can_be_excluded(session_factory, blob_store, embedder, reranker):
    pages = (
        PageText(1, "1 Introduction\n\nCookie parameters are declared with Cookie in the method."),
        PageText(2, "References\n\nCookie, A. Cookie parameters with Cookie. 2020."),
    )
    extracted = ExtractedDocument(title="Paper", pages=pages)
    with session_factory() as session:
        version = index_versions_repo.get_active(session)
        document = Document(
            owner_id=DEV_USER_ID,
            title="Paper",
            filename="paper.pdf",
            mime_type="application/pdf",
            size_bytes=1,
            content_hash="e" * 64,
            storage_key="e" * 64,
            status=DocumentStatus.COMPLETED,
            extracted_text=extracted.text,
            page_count=2,
        )
        document.chunks = build_chunks(extracted, version, embedder)
        session.add(document)
        session.commit()

    def paths(**config):
        result = run(session_factory, embedder, reranker, top_k=5, candidate_k=10, **config)
        return {h.section_path for h in result.hits}

    assert "Paper > References" in paths(mode="hybrid")
    assert "Paper > References" not in paths(mode="hybrid", exclude_references=True)
    assert "Paper > References" not in paths(mode="dense", exclude_references=True)
