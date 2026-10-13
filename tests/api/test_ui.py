"""The chat page is served at /ui with a strict Content-Security-Policy (component 8D)."""

import pytest

pytestmark = pytest.mark.integration


def test_root_redirects_to_the_page(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


@pytest.mark.parametrize(
    "path", ["/ui/", "/ui/app.js", "/ui/render.js", "/ui/styles.css", "/ui/favicon.svg"]
)
def test_page_and_assets_load_without_a_key(client, path):
    client.headers.pop("Authorization")

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_page_may_only_run_its_own_scripts(client):
    csp = client.get("/ui/").headers["Content-Security-Policy"]

    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp


def test_page_has_no_inline_script_or_style(client):
    html = client.get("/ui/").text

    assert "<script>" not in html and "<style>" not in html and "style=" not in html
    assert '<script type="module" src="app.js"></script>' in html


def test_the_api_still_answers_under_its_own_paths(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ui/missing.js").status_code == 404
