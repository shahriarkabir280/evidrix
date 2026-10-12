"""Every migration can be undone and redone (component 6A)."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from tests.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.integration


def test_downgrade_to_base_and_upgrade_to_head(db_engine):
    config = Config("alembic.ini")
    config.attributes["database_url"] = TEST_DATABASE_URL
    config.attributes["configure_logger"] = False

    command.downgrade(config, "base")
    tables_at_base = set(inspect(db_engine).get_table_names()) - {"alembic_version"}
    command.upgrade(config, "head")
    tables_at_head = set(inspect(db_engine).get_table_names())

    assert tables_at_base == set()
    assert {"documents", "chunks", "ingestion_jobs", "api_keys", "chat_queries"} <= tables_at_head
