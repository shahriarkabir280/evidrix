"""FastAPI app factory: lifespan (model loading), routers, middleware.

Component 1A · Plan: Part 1, step 1 and §10.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware import RequestContextMiddleware
from app.api.routes import chat, documents, health, jobs, retrieve
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.embeddings import get_embedder
from app.rag.rerank import get_reranker


def create_app(*, preload_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Here rather than at import: uvicorn sets up its own loggers after importing the app.
        configure_logging(get_settings())
        if preload_models:
            # Load the local models at startup, so the first request isn't the slow one.
            get_embedder()
            get_reranker()
        yield

    app = FastAPI(
        title="Evidrix",
        description="Evidence-grounded RAG knowledge assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(chat.router)
    app.include_router(retrieve.router)
    return app


app = create_app()
