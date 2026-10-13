"""Serves the chat page (app/ui) at /ui, with headers that limit what the page may run.

Component 8D.

The page shows text that comes from documents and from a model that read them. It escapes that
text (app/ui/render.js); the Content-Security-Policy is the second line of defense: only the
page's own scripts and styles may load, it may only talk to this API, and it can't be framed.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class UIFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers.update(SECURITY_HEADERS)
        return response


def install_ui(app: FastAPI) -> None:
    app.mount("/ui", UIFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")
