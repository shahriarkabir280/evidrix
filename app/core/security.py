"""API key generation and hashing.

Component 5C. Why a plain sha256 and a lookup by hash: docs/adr/0004-api-key-storage.md.

A key looks like evx_<43 url-safe characters> (32 random bytes). Only its sha256 is stored, so a
leaked database doesn't leak working keys. The first characters are kept as a non-secret prefix,
to tell keys apart in lists and logs without revealing them.
"""

import hashlib
import secrets

KEY_PREFIX = "evx_"
KEY_RANDOM_BYTES = 32
DISPLAY_PREFIX_LENGTH = len(KEY_PREFIX) + 8
MAX_KEY_LENGTH = 100  # longer input is rejected before hashing


def generate_api_key() -> str:
    # secrets, not random: random is predictable from its earlier outputs.
    return KEY_PREFIX + secrets.token_urlsafe(KEY_RANDOM_BYTES)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def display_prefix(key: str) -> str:
    return key[:DISPLAY_PREFIX_LENGTH]


def looks_like_api_key(value: str) -> bool:
    """A cheap format check, so obvious garbage never reaches the database."""
    return value.startswith(KEY_PREFIX) and len(value) <= MAX_KEY_LENGTH
