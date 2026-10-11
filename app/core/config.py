"""Settings loaded from the environment with pydantic-settings.

Component 1A · Plan: Part 1, step 2.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/evidrix"
    max_upload_bytes: int = 10_000_000
    blob_dir: Path = Path("data/uploads")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge-v1.5 retrieves better when queries (not documents) carry this instruction.
    # The model's own sentence-transformers config leaves the query prompt empty.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    embedding_batch_size: int = 32


@lru_cache
def get_settings() -> Settings:
    return Settings()
