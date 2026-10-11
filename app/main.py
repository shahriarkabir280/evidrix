"""FastAPI app factory: lifespan (model loading), routers, middleware.

Component 1A · Plan: Part 1, step 1 and §10.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, health
from app.rag.embeddings import get_embedder


def create_app(*, preload_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if preload_models:
            # Load the embedding model at startup, so the first request isn't the slow one.
            get_embedder()
        yield

    app = FastAPI(
        title="Evidrix",
        description="Evidence-grounded RAG knowledge assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    return app


app = create_app()
