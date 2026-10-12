"""Users and API keys.

Component 5C.
"""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.security import display_prefix, generate_api_key, hash_api_key
from app.db.models import ApiKey, User

# The user seeded by migration 0001. It owns what the command-line tools ingest (the evaluation
# corpus). The API never falls back to it: every request must authenticate with an API key.
DEV_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def get_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email))


def create_api_key(session: Session, user_id: uuid.UUID, name: str | None = None) -> str:
    """Store a new key for the user and return it. This is the only time the key is visible."""
    key = generate_api_key()
    session.add(
        ApiKey(user_id=user_id, key_hash=hash_api_key(key), prefix=display_prefix(key), name=name)
    )
    session.commit()
    return key


def user_id_for_key(session: Session, key: str) -> uuid.UUID | None:
    """The owner of an active key, or None for an unknown or revoked key.

    The lookup is by hash, so the database compares hashes, never the key itself; see
    docs/adr/0004-api-key-storage.md for why that comparison needn't be constant-time.
    """
    return session.scalar(
        select(ApiKey.user_id).where(
            ApiKey.key_hash == hash_api_key(key), ApiKey.revoked_at.is_(None)
        )
    )


def revoke_api_keys(session: Session, prefix: str) -> int:
    """Revoke the active keys that start with `prefix`; returns how many were revoked."""
    result = session.execute(
        update(ApiKey)
        .where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
        .values(revoked_at=func.now())
    )
    session.commit()
    return result.rowcount
