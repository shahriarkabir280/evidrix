"""GET /health (liveness) and GET /ready (readiness).

Component 1A · /ready added in 6C.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
