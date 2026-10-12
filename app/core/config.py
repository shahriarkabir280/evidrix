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

    redis_url: str = "redis://localhost:6379/0"
    # A job running longer than this is killed by RQ (plan: Part 5, step 7).
    ingestion_job_timeout_seconds: int = 600
    # 1 attempt + 3 retries for transient errors, waiting 10 s, 1 min, 5 min between them.
    ingestion_max_attempts: int = 4
    ingestion_retry_delays_seconds: list[int] = [10, 60, 300]
    # A job still "processing" after this long lost its worker: RQ would have killed it at
    # ingestion_job_timeout_seconds, so the lease must be longer than that.
    ingestion_lease_seconds: int = 900

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge-v1.5 retrieves better when queries (not documents) carry this instruction.
    # The model's own sentence-transformers config leaves the query prompt empty.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    embedding_batch_size: int = 32

    # Any OpenAI-compatible endpoint; the default is a local Ollama server.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores it, but the client requires a value
    llm_model: str = "qwen3-vl:8b-instruct"
    # Must match the server: start Ollama with OLLAMA_CONTEXT_LENGTH=8192.
    llm_context_tokens: int = 8192
    llm_max_output_tokens: int = 512
    # A local 8B model reads ~150 prompt tokens/s; the first call also loads the model.
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 1
    # Evaluation only: a different model family from llm_model grades the answers. Calibration
    # (evals/judge_calibration.jsonl, judge_correctness_v1): gemma3:12b 100%, llama3.1:8b 88%.
    judge_model: str = "gemma3:12b"

    # Cross-encoder that rescores the candidate pool (runs on embedding_device).
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Name of a file in configs/retrieval/. hybrid_rerank won on the dev split (MRR +0.155 over
    # dense, significant). Each chunk sent to the local 8B model adds ~1.8 s of prompt reading,
    # so top_k is also a latency setting.
    retrieval_preset: str = "hybrid_rerank"


@lru_cache
def get_settings() -> Settings:
    return Settings()
