"""Request context: request ID and one access log line per request.

Component 5G.

A plain ASGI middleware rather than Starlette's BaseHTTPMiddleware: it runs in the same task as
the endpoint, so the context it binds (the request ID) is visible to every log line the request
produces, including those from sync endpoints run in the thread pool.
"""

import time

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import REQUEST_ID_HEADER, new_request_id

logger = structlog.get_logger("app.access")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        supplied = headers.get(REQUEST_ID_HEADER.lower().encode(), b"").decode("latin-1")
        request_id = new_request_id(supplied)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500  # if the app fails before sending a response

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((REQUEST_ID_HEADER.encode(), request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            state = scope.get("state", {})
            # The path only: a query string could carry anything, and keys go in headers.
            logger.info(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                user_id=str(state["user_id"]) if "user_id" in state else None,
            )
            structlog.contextvars.clear_contextvars()
