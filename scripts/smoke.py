"""Checks a running deployment end to end, then (optionally) that its data survived a restart.

Component 7E · Plan: Part 7, steps 6-7.

    uv run python -m scripts.smoke --base-url http://localhost:8080 --api-key evx_...
    # restart the services, then:
    uv run python -m scripts.smoke --base-url ... --api-key ... --check-persistence

The smoke run: /health → /ready → upload a fresh document → poll its job until completed
(≤ 120 s) → an answerable question cites that document → an unanswerable one abstains → a
request without a key gets 401. On a retrieval-only deployment (/chat answers 503
chat_disabled), the two questions are checked through /retrieve instead.

The run writes what it created to a state file; --check-persistence reads it and checks that
the document, its chunks and the chat history are still there.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

STATE_FILE = Path(".smoke-state.json")
JOB_TIMEOUT_S = 120
ANSWERABLE = "What colour is the Evidrix smoke-test lighthouse, and when does it blink?"
UNANSWERABLE = "What was the closing price of the Tokyo Stock Exchange yesterday?"


def fixture_document(marker: str) -> bytes:
    """A small document no other user or run has: the marker defeats deduplication."""
    return (
        f"# Smoke test {marker}\n\n"
        "## The lighthouse\n\n"
        "The Evidrix smoke-test lighthouse is painted teal. It blinks every seven seconds, "
        "from dusk until dawn, and its keeper logs every blink in a paper notebook.\n"
    ).encode()


class SmokeFailure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  ok  {message}")


def run(client: httpx.Client) -> dict[str, Any]:
    check(client.get("/health").json() == {"status": "ok"}, "/health answers")
    ready = client.get("/ready")
    check(ready.status_code == 200, f"/ready is ready: {ready.json()}")

    marker = uuid.uuid4().hex[:12]
    upload = client.post(
        "/documents", files={"file": (f"smoke-{marker}.md", fixture_document(marker))}
    )
    check(upload.status_code == 202, f"upload accepted (202), job {upload.json().get('job_id')}")
    document_id, job_id = upload.json()["id"], upload.json()["job_id"]

    deadline = time.monotonic() + JOB_TIMEOUT_S
    while True:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed") or time.monotonic() > deadline:
            break
        time.sleep(2)
    check(job["status"] == "completed", f"job completed within {JOB_TIMEOUT_S} s ({job['status']})")

    chat = client.post("/chat", json={"question": ANSWERABLE, "document_ids": [document_id]})
    if chat.status_code == 503 and chat.json()["error"]["code"] == "chat_disabled":
        print("  --  /chat is disabled here (retrieval-only); checking /retrieve instead")
        hits = client.post(
            "/retrieve", json={"question": ANSWERABLE, "document_ids": [document_id]}
        ).json()["hits"]
        check(bool(hits) and "teal" in hits[0]["text"], "the answer's evidence is retrieved")
    else:
        body = chat.json()
        cited = {c["document_id"] for c in body.get("citations", [])}
        check(chat.status_code == 200 and document_id in cited, "the answer cites the document")
        unanswerable = client.post(
            "/chat", json={"question": UNANSWERABLE, "document_ids": [document_id]}
        ).json()
        check(unanswerable.get("abstained") is True, "an unanswerable question abstains")

    with httpx.Client(base_url=client.base_url, timeout=10) as anonymous:
        no_key = anonymous.get("/documents")
    check(no_key.status_code == 401, "a request without a key is refused (401)")
    return {"document_id": document_id, "chat_enabled": chat.status_code != 503}


def check_persistence(client: httpx.Client, state: dict[str, Any]) -> None:
    document = client.get(f"/documents/{state['document_id']}")
    check(document.status_code == 200, "the document still exists")
    check(document.json()["status"] == "completed", "it is still completed")
    hits = client.post(
        "/retrieve", json={"question": ANSWERABLE, "document_ids": [state["document_id"]]}
    ).json()["hits"]
    check(bool(hits), "its chunks are still searchable")
    if state["chat_enabled"]:
        check(client.get("/chats").json()["total"] > 0, "the chat history is still there")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--check-persistence", action="store_true")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180) as client:
        try:
            if args.check_persistence:
                print(f"Persistence check against {args.base_url}")
                check_persistence(client, json.loads(STATE_FILE.read_text()))
            else:
                print(f"Smoke test against {args.base_url}")
                STATE_FILE.write_text(json.dumps(run(client)))
        except (SmokeFailure, httpx.HTTPError, KeyError) as exc:
            print(f"FAIL {exc!r}")
            sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
