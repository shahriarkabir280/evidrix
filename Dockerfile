# syntax=docker/dockerfile:1
#
# One image for the API and the worker (component 7A). Build stages:
#   builder  - installs the locked dependencies (CPU-only torch on Linux) and downloads the models
#   runtime  - a slim image with the virtualenv, the models and the code, running as a non-root user
#
#   docker build -t evidrix .
#   docker run -p 8000:8000 --env-file .env.docker evidrix                         # API
#   docker run --env-file .env.docker evidrix rq worker --worker-class \
#       rq.worker.SimpleWorker --with-scheduler --url "$REDIS_URL" ingestion       # worker

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change, not on
# every code change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Models at build time, so a container starts without downloading ~200 MB from the internet
# (slow, and a failure point at every deploy). Keep the names in sync with the settings.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV HF_HOME=/models
RUN /app/.venv/bin/python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}', device='cpu'); \
CrossEncoder('${RERANKER_MODEL}', device='cpu')"

COPY alembic.ini ./
COPY migrations ./migrations
COPY configs ./configs
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM ${PYTHON_IMAGE} AS runtime
# UID 1000: Hugging Face Spaces run containers as that user.
RUN useradd --create-home --uid 1000 evidrix
WORKDIR /app
COPY --from=builder --chown=evidrix:evidrix /app /app
COPY --from=builder --chown=evidrix:evidrix /models /models
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    BLOB_DIR=/app/data/uploads
USER evidrix
EXPOSE 8000
# exec: uvicorn becomes PID 1 and gets the platform's SIGTERM for a graceful shutdown.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
