"""FastAPI app factory: lifespan (model loading), routers, middleware.

Component 1A · Plan: Part 1, step 1 and §10.
"""

from fastapi import FastAPI

from app.api.routes import health


def create_app() -> FastAPI:
    app = FastAPI(
        title="Evidrix",
        description="Evidence-grounded RAG knowledge assistant",
        version="0.1.0",
    )
    app.include_router(health.router)
    return app


app = create_app()
