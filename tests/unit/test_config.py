import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_to_development(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_env == "development"


def test_reads_app_env_from_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    settings = Settings(_env_file=None)

    assert settings.app_env == "production"


def test_rejects_unknown_app_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "staging")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
