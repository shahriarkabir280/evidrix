"""PostgresBlobStore behaves like LocalBlobStore (component 7B)."""

import pytest

from app.core.config import Settings
from app.storage.blob import (
    BlobNotFoundError,
    LocalBlobStore,
    PostgresBlobStore,
    content_key,
    make_blob_store,
)

pytestmark = pytest.mark.integration

DATA = b"%PDF-1.7 pretend document bytes"
NOTE = b"# Notes\n\nThe bytes of this upload are stored in Postgres, not on disk.\n"


@pytest.fixture
def blob_store(session_factory):
    """Overrides the conftest fixture, so the API client and the worker use Postgres too."""
    return PostgresBlobStore(session_factory)


@pytest.fixture
def store(blob_store):
    return blob_store


def test_put_then_get_returns_the_same_bytes(store):
    key = content_key(DATA)

    store.put(key, DATA)

    assert store.get(key) == DATA


def test_putting_the_same_key_twice_is_harmless(store):
    key = content_key(DATA)

    store.put(key, DATA)
    store.put(key, DATA)

    assert store.get(key) == DATA


def test_get_missing_blob_raises(store):
    with pytest.raises(BlobNotFoundError):
        store.get(content_key(b"never stored"))


def test_delete_removes_the_blob_and_tolerates_missing_ones(store):
    key = content_key(DATA)
    store.put(key, DATA)

    store.delete(key)
    store.delete(key)

    with pytest.raises(BlobNotFoundError):
        store.get(key)


@pytest.mark.parametrize("bad_key", ["../../etc/passwd", "ABC", "", "a" * 63, "g" * 64])
def test_keys_that_are_not_sha256_hex_are_rejected(store, bad_key):
    with pytest.raises(ValueError):
        store.put(bad_key, DATA)


def test_upload_and_ingestion_work_with_blobs_in_postgres(client, run_worker, store):
    response = client.post("/documents", files={"file": ("a.md", NOTE)})
    run_worker()

    document = client.get(f"/documents/{response.json()['id']}").json()
    assert document["status"] == "completed"
    assert store.get(document["content_hash"]) == NOTE


def test_the_setting_chooses_the_backend(tmp_path):
    local = make_blob_store(Settings(_env_file=None, blob_dir=tmp_path))
    postgres = make_blob_store(Settings(_env_file=None, blob_backend="postgres"))

    assert isinstance(local, LocalBlobStore)
    assert isinstance(postgres, PostgresBlobStore)
