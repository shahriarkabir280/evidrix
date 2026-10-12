"""structlog JSON logging and the request-ID contextvar.

Component 5G · Plan: Part 5, step 9.

One JSON object per line, so logs can be searched by field (request_id, job_id, status) instead
of by text. Both structlog loggers and plain `logging` loggers (uvicorn, RQ, libraries) go
through the same processors, and every line gets the context bound with bind_contextvars, such
as the request ID, without passing it around.

What is never logged: API keys, document text, questions and answers. A question is logged as its
length and a hash, which is enough to spot repeats without storing what people asked.
"""

import hashlib
import logging
import re
import sys
import uuid
from typing import TextIO

import structlog

from app.core.config import Settings

HANDLER_NAME = "evidrix"
REQUEST_ID_HEADER = "X-Request-ID"
# A client-supplied ID is reused only if it is short and plain, so it can't inject log content.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

_SHARED_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.ExtraAdder(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def configure_logging(settings: Settings, stream: TextIO | None = None) -> None:
    """Route all logging through structlog's formatter. Safe to call more than once."""
    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    root = logging.getLogger()
    # Replace only our own handler, so pytest's capture handlers keep working.
    root.handlers = [h for h in root.handlers if h.get_name() != HANDLER_NAME]
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    # The request middleware writes one access line per request, with the request ID.
    logging.getLogger("uvicorn.access").disabled = True


def new_request_id(supplied: str | None = None) -> str:
    if supplied and _SAFE_REQUEST_ID.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")


def text_fingerprint(text: str) -> dict[str, object]:
    """What to log instead of a user's text: its length and a short hash."""
    return {"chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
