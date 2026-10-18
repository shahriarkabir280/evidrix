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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True  # false: one readable line per event, for a terminal
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/evidrix"
    max_upload_bytes: int = 10_000_000
    blob_dir: Path = Path("data/uploads")
    # "postgres" on hosts whose container file system doesn't survive a restart.
    blob_backend: Literal["local", "postgres"] = "local"

    redis_url: str = "redis://localhost:6379/0"
    # "rq": a separate worker process takes jobs from Redis (make worker). "inline": the API runs
    # them in a background thread, for a single-instance deployment without a worker
    # (app/worker/inline.py has the trade-offs).
    ingestion_mode: Literal["rq", "inline"] = "rq"
    # False for a retrieval-only deployment without an LLM: /chat answers 503 chat_disabled.
    chat_enabled: bool = True

    # Requests per user per minute (fixed window). Each /chat costs a model call, so it is the
    # tightest; /retrieve is cheap; uploads cost embedding time in the worker.
    rate_limit_chat_per_minute: int = 20
    rate_limit_retrieve_per_minute: int = 60
    rate_limit_upload_per_minute: int = 10
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
    # A version in app/rag/prompts. answer_v2 adds a reminder after the sources: it stopped an
    # injected "citation rule" that v1 followed, with no measurable change on the benchmarks
    # (docs/security.md).
    answer_prompt_version: str = "answer_v2"
    # Per model call. Measured on the dev split: generation p95 is ~25 s warm, and loading the
    # model on the first call adds 10-20 s. Worst case for one /chat: 2 calls (one retry for
    # unusable output) x 2 attempts (llm_max_retries) x 60 s = 4 min.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 1
    # Calls in flight at once from this process: match the model server's parallelism
    # (OLLAMA_NUM_PARALLEL, 1 by default). Others wait up to llm_queue_timeout_seconds for a
    # slot, then get 503 llm_busy. See OpenAICompatibleProvider for the measurement.
    llm_max_concurrency: int = 1
    llm_queue_timeout_seconds: float = 120.0
    # json_schema (Ollama, most hosted models) | json_object (models with JSON mode only) | none.
    llm_response_format: Literal["json_schema", "json_object", "none"] = "json_schema"
    # Evaluation only: a different model family from llm_model grades the answers. Calibration
    # (evals/judge_calibration.jsonl, judge_correctness_v1): gemma3:12b 100%, llama3.1:8b 88%.
    judge_model: str = "gemma3:12b"
    judge_timeout_seconds: float = 120.0  # a 12B model grading long answers, offline

    # Cross-encoder that rescores the candidate pool (runs on embedding_device).
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Pairs scored per forward pass. Peak memory grows with it: the attention matrices of 20
    # pairs of 512 tokens take ~250 MB. Smaller batches trade a little speed for memory on a
    # small instance (measured in docs/deploy.md).
    reranker_batch_size: int = 32

    # Name of a file in configs/retrieval/. hybrid_rerank won on the dev split (MRR +0.155 over
    # dense, significant). Each chunk sent to the local 8B model adds ~1.8 s of prompt reading,
    # so top_k is also a latency setting.
    retrieval_preset: str = "hybrid_rerank"


@lru_cache
def get_settings() -> Settings:
    return Settings()
