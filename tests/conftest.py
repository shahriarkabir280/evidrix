"""Shared fixtures: test database, migrations, app client, dependency overrides.

Component 1E.
"""

import os
import uuid

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.api.deps import get_blob_store, get_db, get_settings
from app.core.config import Settings
from app.db.models import User
from app.main import create_app
from app.storage.blob import LocalBlobStore

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5433/evidrix_test"
)
# Small limit so the "too large" test doesn't need to build a 10 MB upload.
TEST_MAX_UPLOAD_BYTES = 50_000


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(TEST_DATABASE_URL)
    try:
        engine.connect().close()
    except OperationalError:
        pytest.fail(f"Test database is not reachable at {TEST_DATABASE_URL}. Run `make up`.")

    # Apply the real migrations, so tests exercise the same schema as production.
    alembic_config = Config("alembic.ini")
    alembic_config.attributes["database_url"] = TEST_DATABASE_URL
    alembic_config.attributes["configure_logger"] = False
    command.upgrade(alembic_config, "head")

    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(db_engine):
    # Start every test from empty document tables (the seeded dev user stays).
    with db_engine.begin() as connection:
        connection.execute(text("TRUNCATE document_pages, documents"))
        connection.execute(text("DELETE FROM users WHERE email LIKE '%@test.local'"))
    return sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest.fixture
def db_session(session_factory):
    with session_factory() as session:
        yield session


@pytest.fixture
def blob_store(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def client(session_factory, blob_store):
    settings = Settings(
        _env_file=None, database_url=TEST_DATABASE_URL, max_upload_bytes=TEST_MAX_UPLOAD_BYTES
    )

    def override_get_db():
        with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_blob_store] = lambda: blob_store
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def other_user(db_session):
    user = User(id=uuid.uuid4(), email=f"{uuid.uuid4().hex[:8]}@test.local")
    db_session.add(user)
    db_session.commit()
    return user
