"""BlobStore protocol: LocalBlobStore (files) and PostgresBlobStore (a bytea table).

Component 1C · Postgres store in 7B.

A container's file system is ephemeral on most hosts, so a deployment keeps the uploaded bytes
in Postgres, next to the documents (BLOB_BACKEND=postgres). At this scale (files up to 10 MB,
tens of MB in total) that avoids running an object store; an S3 store would implement the same
three methods if the volume grew.
"""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from app.core.config import Settings

# Keys are sha256 hex digests, which also rules out path tricks like "../".
_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")


class BlobNotFoundError(Exception):
    pass


def content_key(data: bytes) -> str:
    """Content-addressed key: identical bytes always map to the same key."""
    return hashlib.sha256(data).hexdigest()


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class LocalBlobStore:
    """Stores each blob as a file under root, sharded by the first two hex characters."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file, then rename: readers never see a half-written blob.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise BlobNotFoundError(key) from exc

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / _checked(key)


class PostgresBlobStore:
    """Stores each blob as a row in the `blobs` table, in its own short transaction."""

    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._sessions = session_factory

    def put(self, key: str, data: bytes) -> None:
        from sqlalchemy.dialects.postgresql import insert

        from app.db.models import Blob

        with self._sessions() as session:
            # Content-addressed: a key that exists already holds these exact bytes.
            session.execute(
                insert(Blob).values(key=_checked(key), data=data).on_conflict_do_nothing()
            )
            session.commit()

    def get(self, key: str) -> bytes:
        from app.db.models import Blob

        with self._sessions() as session:
            blob = session.get(Blob, _checked(key))
            if blob is None:
                raise BlobNotFoundError(key)
            return blob.data

    def delete(self, key: str) -> None:
        from sqlalchemy import delete

        from app.db.models import Blob

        with self._sessions() as session:
            session.execute(delete(Blob).where(Blob.key == _checked(key)))
            session.commit()


def make_blob_store(settings: "Settings") -> BlobStore:
    if settings.blob_backend == "postgres":
        from app.db.session import get_session_factory

        return PostgresBlobStore(get_session_factory())
    return LocalBlobStore(settings.blob_dir)


def _checked(key: str) -> str:
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError(f"Invalid blob key: {key!r}")
    return key
