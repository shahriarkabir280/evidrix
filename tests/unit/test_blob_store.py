import hashlib

import pytest

from app.storage.blob import BlobNotFoundError, LocalBlobStore, content_key

DATA = b"%PDF-1.7 pretend document bytes"


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path)


def test_content_key_is_the_sha256_of_the_bytes():
    assert content_key(DATA) == hashlib.sha256(DATA).hexdigest()


def test_put_then_get_returns_the_same_bytes(store):
    key = content_key(DATA)

    store.put(key, DATA)

    assert store.get(key) == DATA


def test_blobs_are_sharded_by_key_prefix(store, tmp_path):
    key = content_key(DATA)

    store.put(key, DATA)

    assert (tmp_path / key[:2] / key).is_file()


def test_put_leaves_no_temp_files_behind(store, tmp_path):
    key = content_key(DATA)

    store.put(key, DATA)

    assert [p.name for p in (tmp_path / key[:2]).iterdir()] == [key]


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
