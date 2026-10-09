"""BlobStore protocol: LocalBlobStore now, S3BlobStore for deployment.

Component 1C · S3 in 7B.
"""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

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
        if not _KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid blob key: {key!r}")
        return self.root / key[:2] / key
