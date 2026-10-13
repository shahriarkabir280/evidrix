"""Creating and switching index versions (component 8A, chunk-size experiment)."""

import pytest
from sqlalchemy import func, select

from app.db.models import IndexVersion
from app.db.repositories import index_versions as index_versions_repo
from tests.conftest import TEST_INDEX_VERSION

pytestmark = pytest.mark.integration


def test_new_version_copies_the_embedding_model_and_starts_inactive(db_session):
    version = index_versions_repo.create_like_active(
        db_session, "test-c20", {"chunk_size_tokens": 20}
    )
    active = index_versions_repo.get_active(db_session)

    assert active.name == TEST_INDEX_VERSION
    assert (version.embedding_model, version.is_active) == (active.embedding_model, False)
    assert version.chunking == {**active.chunking, "chunk_size_tokens": 20}


def test_activating_leaves_exactly_one_active_version(db_session):
    index_versions_repo.create_like_active(db_session, "test-c20", {"chunk_size_tokens": 20})

    index_versions_repo.activate(db_session, "test-c20")

    active_count = db_session.scalar(
        select(func.count()).select_from(IndexVersion).where(IndexVersion.is_active)
    )
    assert active_count == 1
    assert index_versions_repo.get_active(db_session).name == "test-c20"


def test_activating_an_unknown_version_changes_nothing(db_session):
    with pytest.raises(index_versions_repo.NoActiveIndexVersionError):
        index_versions_repo.activate(db_session, "test-missing")

    assert index_versions_repo.get_active(db_session).name == TEST_INDEX_VERSION
