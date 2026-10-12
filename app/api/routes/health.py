"""GET /health (liveness) and GET /ready (readiness).

Component 1A · /ready added in 6C.

Liveness asks "is the process alive?": a failure means restart it. It checks nothing else,
because restarting a healthy process won't fix a database outage.

Readiness asks "should this instance get traffic?": the database and Redis must answer. Its main
use is a new instance during a deploy: one with a wrong DATABASE_URL or REDIS_URL never gets
traffic. The models are loaded and warmed up before the server accepts requests at all (see the
lifespan in app/main.py). The LLM is not checked: every instance shares it, so taking instances
out of rotation wouldn't help, and /chat already answers 503 when it is down.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from redis import Redis
from sqlalchemy import text

from app.api.deps import DbSession, get_redis
from app.api.errors import error_body

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", responses={503: {"description": "A dependency is unreachable"}})
def ready(
    request: Request, session: DbSession, redis: Annotated[Redis, Depends(get_redis)]
) -> JSONResponse:
    checks = {"database": "ok", "redis": "ok"}
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        logger.warning("readiness_check_failed", dependency="database", exc_info=True)
        checks["database"] = "unreachable"
    try:
        redis.ping()
    except Exception:
        logger.warning("readiness_check_failed", dependency="redis", exc_info=True)
        checks["redis"] = "unreachable"
    checks["models"] = "warmed_up" if getattr(request.app.state, "warmed_up", False) else "lazy"

    failed = [name for name, state in checks.items() if state == "unreachable"]
    if failed:
        body = error_body("not_ready", f"Unreachable: {', '.join(failed)}.", checks=checks)
        return JSONResponse(body, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return JSONResponse({"status": "ready", "checks": checks})
