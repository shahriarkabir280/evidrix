"""Error envelope and exception handlers.

Component 6C · Plan: §4 (error envelope), Part 6, step 5.

Every 4xx/5xx answers in one shape, so a client needs one code path for errors:

    {"error": {"code": "document_not_found", "message": "...", "request_id": "..."}}

`code` is stable and meant for programs; `message` is for people and may change. The request ID
lets a user's bug report be matched with the server's log lines.

Unhandled exceptions (500) are not handled here but in RequestContextMiddleware: Starlette runs
a handler for `Exception` outside all middleware, where the request ID is already gone.
"""

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import current_request_id


def error_body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "request_id": current_request_id(), **extra}
    }


def _code_for(status_code: int) -> str:
    """e.g. 405 -> "method_not_allowed", for errors raised by the framework itself."""
    return HTTPStatus(status_code).phrase.lower().replace(" ", "_").replace("-", "_")


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        body = error_body(exc.detail["code"], exc.detail.get("message", ""))
    else:  # raised by FastAPI/Starlette, e.g. an unknown path
        body = error_body(_code_for(exc.status_code), str(exc.detail))
    return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
        for error in exc.errors()
    ]
    message = "; ".join(f"{d['field']}: {d['message']}" for d in details)
    return JSONResponse(error_body("validation_error", message, details=details), status_code=422)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
