"""Asks one question and prints the answer, citations, and retrieved chunks.

Component 2G · API key in 5C. Talks to a running API (make dev), so it exercises the real
endpoint. The key comes from EVIDRIX_API_KEY (create one with `app.cli create-key`).

    uv run python -m app.cli create-key --email dev@evidrix.local   # once
    export EVIDRIX_API_KEY=evx_...
    uv run python -m scripts.ask "How do I declare a cookie parameter?"
    uv run python -m scripts.ask "..." --doc <document-id> --doc <another-id>
"""

import argparse
import json
import os
import textwrap
import urllib.error
import urllib.request


def ask(base_url: str, api_key: str, question: str, document_ids: list[str] | None) -> dict:
    body = {"question": question}
    if document_ids:
        body["document_ids"] = document_ids
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {base_url} ({exc.reason}). Is `make dev` running?") from exc


def where(item: dict) -> str:
    if item["section_path"]:
        return item["section_path"]
    if item["page_start"] is None:
        return ""
    if item["page_end"] in (None, item["page_start"]):
        return f"page {item['page_start']}"
    return f"pages {item['page_start']}-{item['page_end']}"


def show(result: dict) -> None:
    status = f"abstained ({result['abstain_reason']})" if result["abstained"] else "answered"
    print(f"\n{status}\n")
    print(textwrap.fill(result["answer"], 100))
    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  [{c['marker']}] {c['document_title']} · {where(c)}")
    print("\nRetrieved:")
    for r in result["retrieved"]:
        print(f"  {r['rank']}. [{r['score']:.3f}] {r['document_title']} · {where(r)}")
    timings = ", ".join(f"{name} {ms / 1000:.2f}s" for name, ms in result["timings_ms"].items())
    tokens = (
        f"{result['usage']['prompt_tokens']} prompt + {result['usage']['completion_tokens']} output"
    )
    print(f"\n{timings} · {tokens} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--doc", action="append", dest="document_ids", help="limit to a document")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    api_key = os.environ.get("EVIDRIX_API_KEY")
    if not api_key:
        raise SystemExit("Set EVIDRIX_API_KEY (uv run python -m app.cli create-key --email ...).")
    show(ask(args.url, api_key, args.question, args.document_ids))


if __name__ == "__main__":
    main()
