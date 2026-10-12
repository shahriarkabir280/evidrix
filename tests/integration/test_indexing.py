import numpy as np
import pytest
from sqlalchemy import func, select, text

from app.db.models import Chunk, Document, DocumentStatus
from app.db.repositories import index_versions as index_versions_repo
from app.db.repositories.users import DEV_USER_ID
from app.rag.extract import DocumentKind, extract
from app.services import ingestion
from tests.factories import make_pdf

pytestmark = pytest.mark.integration

GUIDE = (
    "# Cookie Parameters\n\n"
    "You can define Cookie parameters the same way you define Query and Path parameters.\n\n"
    "## Import Cookie\n\n"
    "First import Cookie from fastapi, then declare the parameter with a default value.\n\n"
    "## Recap\n\n"
    "Declare cookies with Cookie, using the same pattern as Query and Path.\n"
)


def ingest(session, blob_store, embedder, filename, data):
    return ingestion.ingest_upload(
        session, blob_store, embedder, owner_id=DEV_USER_ID, filename=filename, data=data
    ).document


def test_upload_stores_chunks_with_embeddings(session_factory, blob_store, embedder):
    with session_factory() as session:
        document = ingest(session, blob_store, embedder, "cookies.md", GUIDE.encode())
        chunks = document.chunks
        active = index_versions_repo.get_active(session)

    assert document.status is DocumentStatus.COMPLETED
    assert len(chunks) == 3  # one per "##" section
    assert [c.section_path for c in chunks] == [
        "Cookie Parameters",
        "Cookie Parameters > Import Cookie",
        "Cookie Parameters > Recap",
    ]
    for chunk in chunks:
        assert chunk.index_version_id == active.id
        assert document.extracted_text[chunk.char_start : chunk.char_end] == chunk.text
        assert len(chunk.embedding) == 384
        assert np.linalg.norm(chunk.embedding) == pytest.approx(1.0, abs=1e-5)


def test_embedding_includes_the_section_header_but_stored_text_does_not(
    session_factory, blob_store, embedder
):
    with session_factory() as session:
        recap = ingest(session, blob_store, embedder, "cookies.md", GUIDE.encode()).chunks[-1]

    expected = embedder.embed_documents([f"Cookie Parameters > Recap\n\n{recap.text}"])[0]
    assert recap.text.startswith("## Recap")
    assert np.allclose(recap.embedding, expected, atol=1e-6)


def test_pdf_chunks_carry_page_ranges(session_factory, blob_store, embedder):
    pages = [f"Page {n} explains logging handlers and formatters in detail." for n in (1, 2, 3)]

    with session_factory() as session:
        chunks = ingest(session, blob_store, embedder, "logging.pdf", make_pdf(pages)).chunks

    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3


def test_embedding_failure_leaves_no_chunks(session_factory, blob_store, embedder, monkeypatch):
    def broken(texts):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(embedder, "embed_documents", broken)

    with session_factory() as session:
        document = ingest(session, blob_store, embedder, "cookies.md", GUIDE.encode())
        chunk_count = session.scalar(select(func.count()).select_from(Chunk))

    assert document.status is DocumentStatus.FAILED
    assert document.error_code == "internal_error"
    assert chunk_count == 0


def test_embedder_that_does_not_match_the_index_version_is_refused(
    session_factory, blob_store, embedder
):
    with session_factory() as session:
        # Make the seeded bge index version active while the loaded embedder is the fake one.
        session.execute(text("UPDATE index_versions SET is_active = false"))
        session.execute(
            text("UPDATE index_versions SET is_active = true WHERE name = 'bge-small_c400_o64'")
        )
        session.commit()

        with pytest.raises(ingestion.IndexConfigurationError):
            ingest(session, blob_store, embedder, "cookies.md", GUIDE.encode())
        document_count = session.scalar(select(func.count()).select_from(Document))

    assert document_count == 0


def test_stored_extraction_rebuilds_pdf_pages_exactly(session_factory, blob_store, embedder):
    data = make_pdf(["First page about sorting.", "", "Third page about sorting keys."])
    original = extract(data, "sorting.pdf", DocumentKind.PDF)

    with session_factory() as session:
        document = ingest(session, blob_store, embedder, "sorting.pdf", data)
        rebuilt = ingestion.stored_extraction(document)

    assert rebuilt == original


def test_reindex_rebuilds_chunks_without_duplicates(session_factory, blob_store, embedder):
    with session_factory() as session:
        document = ingest(session, blob_store, embedder, "cookies.md", GUIDE.encode())
        original = [(c.chunk_index, c.text) for c in document.chunks]
        session.execute(text("DELETE FROM chunks"))
        session.commit()
        session.expire_all()

        first = ingestion.reindex_document(session, embedder, document)
        second = ingestion.reindex_document(session, embedder, document)
        stored = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()

    assert first == second == len(original)
    assert [(c.chunk_index, c.text) for c in stored] == original
