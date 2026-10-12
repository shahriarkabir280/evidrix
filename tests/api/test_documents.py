import uuid

import pytest
from sqlalchemy import func, select

from app.db.models import Document, DocumentPage
from app.storage.blob import content_key
from tests.conftest import TEST_MAX_UPLOAD_BYTES
from tests.factories import make_image_only_pdf, make_pdf

pytestmark = pytest.mark.integration

MARKDOWN = b"# Dependencies with yield\n\nFastAPI supports dependencies that clean up after use.\n"


def upload(client, filename: str, data: bytes):
    return client.post("/documents", files={"file": (filename, data)})


def upload_and_process(client, ingest, filename: str, data: bytes) -> dict:
    """Upload, let the worker run the job, and return the document as the API now shows it."""
    return client.get(f"/documents/{ingest(filename, data)}").json()


def document_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(Document))


# --- Successful uploads ------------------------------------------------------------------


def test_upload_is_accepted_and_queued_without_processing(client, queue):
    response = upload(client, "deps.md", MARKDOWN)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["title"] == "deps"  # the real title is known only after extraction
    assert body["duplicate"] is False
    assert queue.job_ids == [uuid.UUID(body["job_id"])]


def test_markdown_upload_is_extracted_by_the_worker(client, ingest):
    document = upload_and_process(client, ingest, "deps.md", MARKDOWN)

    assert document["status"] == "completed"
    assert document["title"] == "Dependencies with yield"
    assert document["mime_type"] == "text/markdown"
    assert document["page_count"] is None


def test_pdf_upload_records_pages(client, ingest, db_session):
    pdf = make_pdf(["Page one is about path parameters.", "Page two is about query params."])

    body = upload_and_process(client, ingest, "params.pdf", pdf)

    assert body["page_count"] == 2
    document = db_session.get(Document, uuid.UUID(body["id"]))
    pages = db_session.scalars(select(DocumentPage).order_by(DocumentPage.page_number)).all()
    assert [page.page_number for page in pages] == [1, 2]
    assert "query params" in document.extracted_text[pages[1].char_start : pages[1].char_end]


def test_upload_stores_original_bytes_under_content_hash(client, blob_store):
    upload(client, "deps.md", MARKDOWN)

    assert blob_store.get(content_key(MARKDOWN)) == MARKDOWN


def test_client_supplied_path_is_reduced_to_base_name(client):
    response = upload(client, "../../etc/deps.md", MARKDOWN)

    assert response.json()["filename"] == "deps.md"


# --- Deduplication -----------------------------------------------------------------------


def test_reupload_returns_existing_document_and_queues_nothing(client, queue, db_session):
    first = upload(client, "deps.md", MARKDOWN)

    second = upload(client, "renamed-copy.md", MARKDOWN)

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["duplicate"] is True
    assert document_count(db_session) == 1
    assert len(queue.job_ids) == 1


# --- Rejected uploads (no document row is created) ---------------------------------------


@pytest.mark.parametrize(
    ("filename", "data", "status_code", "code"),
    [
        ("tool.exe", b"MZ executable", 415, "unsupported_file_type"),
        ("fake.pdf", b"just text pretending to be a pdf", 415, "file_type_mismatch"),
        ("binary.txt", b"MZ\x00\x00\x00binary", 415, "file_type_mismatch"),
        ("empty.md", b"", 422, "empty_file"),
        ("huge.txt", b"a" * (TEST_MAX_UPLOAD_BYTES + 1), 413, "file_too_large"),
    ],
)
def test_invalid_uploads_are_rejected(client, queue, db_session, filename, data, status_code, code):
    response = upload(client, filename, data)

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert document_count(db_session) == 0
    assert queue.job_ids == []


def test_request_without_file_is_a_validation_error(client):
    assert client.post("/documents").status_code == 422


# --- Extraction failures are recorded, never marked completed ----------------------------


@pytest.mark.parametrize(
    ("filename", "data", "error_code"),
    [
        ("scan.pdf", make_image_only_pdf(), "no_extractable_text"),
        ("broken.pdf", b"%PDF-1.7\nnot really a pdf body", "invalid_pdf"),
        ("latin1.txt", b"caf\xe9 au lait, encoded as latin-1 instead of utf-8", "invalid_encoding"),
    ],
)
def test_extraction_failure_is_recorded(client, ingest, filename, data, error_code):
    body = upload_and_process(client, ingest, filename, data)

    assert body["status"] == "failed"
    assert body["error_code"] == error_code
    assert body["error_message"]


# --- Listing and fetching ----------------------------------------------------------------


def test_list_returns_newest_first_with_pagination(client, ingest):
    ingest("first.md", b"# First\n\nThe first document body text.")
    ingest("second.md", b"# Second\n\nThe second document body text.")

    page = client.get("/documents", params={"limit": 1}).json()

    assert page["total"] == 2
    assert [item["title"] for item in page["items"]] == ["Second"]


def test_unavailable_queue_fails_the_upload_visibly(client, queue, db_session):
    queue.available = False

    response = upload(client, "deps.md", MARKDOWN)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "queue_unavailable"
    # The document exists but is marked failed, never left looking queued forever.
    document = db_session.scalar(select(Document))
    assert (document.status, document.error_code) == ("failed", "queue_unavailable")


def test_failed_document_is_retried_by_uploading_it_again(client, queue, run_worker):
    queue.available = False
    upload(client, "deps.md", MARKDOWN)  # 503: the document is recorded as failed
    queue.available = True

    again = upload(client, "deps.md", MARKDOWN)
    run_worker()

    assert again.status_code == 202
    body = again.json()
    assert (body["duplicate"], body["status"], body["error_code"]) == (True, "queued", None)
    assert queue.job_ids == []  # the worker ran the new job
    assert client.get(f"/documents/{body['id']}").json()["status"] == "completed"
    assert client.get(f"/jobs/{body['job_id']}").json()["status"] == "completed"


def test_get_document_by_id(client):
    created = upload(client, "deps.md", MARKDOWN).json()

    response = client.get(f"/documents/{created['id']}")

    assert response.status_code == 200
    assert response.json()["content_hash"] == created["content_hash"]


def test_unknown_document_is_not_found(client):
    response = client.get(f"/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_malformed_document_id_is_a_validation_error(client):
    assert client.get("/documents/not-a-uuid").status_code == 422


def test_other_users_documents_are_invisible(client, db_session, other_user):
    foreign = Document(
        owner_id=other_user.id,
        title="Private",
        filename="private.md",
        mime_type="text/markdown",
        size_bytes=10,
        content_hash="f" * 64,
        storage_key="f" * 64,
    )
    db_session.add(foreign)
    db_session.commit()

    assert client.get("/documents").json()["total"] == 0
    assert client.get(f"/documents/{foreign.id}").status_code == 404
