"""Shared fixtures: test database, migrations, app client, dependency overrides.

Component 1E.
"""

import os
import uuid

import fakeredis
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.api.deps import (
    get_blob_store,
    get_db,
    get_embedder,
    get_ingestion_queue,
    get_llm,
    get_reranker,
    get_settings,
)
from app.api.rate_limit import RateLimiter, get_rate_limiter
from app.core.config import Settings
from app.db.models import User
from app.db.repositories import users as users_repo
from app.db.repositories.users import DEV_USER_ID
from app.main import create_app
from app.services.ingestion import run_ingestion_job
from app.storage.blob import LocalBlobStore
from tests.fakes import FakeEmbedder, FakeLLM, FakeQueue, FakeReranker

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5433/evidrix_test"
)
# Small limit so the "too large" test doesn't need to build a 10 MB upload.
TEST_MAX_UPLOAD_BYTES = 50_000
# Tests index with FakeEmbedder (word-count tokens), so chunks are small and easy to reason about.
TEST_INDEX_VERSION = "test-fake"
TEST_CHUNKING = (
    '{"chunk_size_tokens": 40, "max_tokens": 50, "overlap_tokens": 10, '
    '"min_chunk_tokens": 10, "section_level": 2}'
)


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
    # Start every test from empty document tables (the seeded dev user stays) and with an
    # active index version that matches FakeEmbedder.
    with db_engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE chat_queries, api_keys, ingestion_jobs, chunks, document_pages, documents"
            )
        )
        connection.execute(text("DELETE FROM users WHERE email LIKE '%@test.local'"))
        connection.execute(text("DELETE FROM index_versions WHERE name LIKE 'test-%'"))
        connection.execute(text("UPDATE index_versions SET is_active = false"))
        connection.execute(
            text(
                "INSERT INTO index_versions "
                "(name, embedding_model, embedding_dim, chunking, contextual_header, is_active) "
                "VALUES (:name, :model, :dim, CAST(:chunking AS jsonb), true, true)"
            ),
            {
                "name": TEST_INDEX_VERSION,
                "model": FakeEmbedder.model_name,
                "dim": FakeEmbedder.dim,
                "chunking": TEST_CHUNKING,
            },
        )
    return sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def reranker():
    return FakeReranker()


@pytest.fixture
def llm():
    """Abstains by default; tests that need an answer override get_llm with their own FakeLLM."""
    return FakeLLM()


@pytest.fixture
def db_session(session_factory):
    with session_factory() as session:
        yield session


@pytest.fixture
def blob_store(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
def run_worker(queue, session_factory, blob_store, embedder):
    """Process every job the API has enqueued so far, as the RQ worker would."""

    def run() -> None:
        while queue.job_ids:
            with session_factory() as session:
                run_ingestion_job(session, blob_store, embedder, queue.job_ids.pop(0))

    return run


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """The real limiter on an in-memory Redis, fresh for every test."""
    return RateLimiter(fakeredis.FakeRedis())


@pytest.fixture
def api_key(db_session) -> str:
    """A key for the seeded dev user, who owns what the service-level tests ingest."""
    return users_repo.create_api_key(db_session, DEV_USER_ID, "tests")


@pytest.fixture
def client(session_factory, blob_store, embedder, reranker, llm, queue, api_key, rate_limiter):
    """Authenticated as the dev user. Pass headers={"Authorization": ...} to act as someone else."""
    settings = Settings(
        _env_file=None, database_url=TEST_DATABASE_URL, max_upload_bytes=TEST_MAX_UPLOAD_BYTES
    )

    def override_get_db():
        with session_factory() as session:
            yield session

    app = create_app(preload_models=False)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_blob_store] = lambda: blob_store
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_queue] = lambda: queue
    app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
    with TestClient(app, headers={"Authorization": f"Bearer {api_key}"}) as test_client:
        yield test_client


@pytest.fixture
def other_user(db_session):
    user = User(id=uuid.uuid4(), email=f"{uuid.uuid4().hex[:8]}@test.local")
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def ingest(client, run_worker):
    """Upload a file through the API and process it; returns the document id."""

    def upload_and_process(filename: str, data: bytes) -> str:
        response = client.post("/documents", files={"file": (filename, data)})
        assert response.status_code == 202, response.text
        run_worker()
        return response.json()["id"]

    return upload_and_process
