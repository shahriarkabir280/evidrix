"""FastAPI app factory: lifespan (model loading), routers, middleware.

Component 1A · Plan: Part 1, step 1 and §10.
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.errors import install_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import chat, documents, health, jobs, retrieve
from app.api.ui import install_ui
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.embeddings import get_embedder
from app.rag.rerank import get_reranker
from app.worker.queue import get_ingestion_queue


def create_app(*, preload_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Here rather than at import: uvicorn sets up its own loggers after importing the app.
        configure_logging(get_settings())
        app.state.warmed_up = False
        if preload_models:
            # Load the models and run each once before accepting requests: the first call
            # also initializes the device (MPS) and its kernels, which loading alone doesn't.
            started = time.perf_counter()
            get_embedder().embed_query("warm-up")
            get_reranker().score("warm-up", ["warm-up"])
            app.state.warmed_up = True
            structlog.get_logger(__name__).info(
                "models_warmed_up", duration_ms=round((time.perf_counter() - started) * 1000, 1)
            )
            if get_settings().ingestion_mode == "inline":
                # No worker to restart in this mode: pick up what the last process left.
                get_ingestion_queue().recover()  # type: ignore[attr-defined]
        yield

    app = FastAPI(
        title="Evidrix",
        description="Evidence-grounded RAG knowledge assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(chat.router)
    app.include_router(retrieve.router)
    install_ui(app)  # last: /ui must not shadow an API route
    return app


app = create_app()
