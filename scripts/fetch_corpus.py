"""Downloads the pinned evaluation corpus and writes the manifest.

Component 2A.

    uv run python -m scripts.fetch_corpus           # download, normalize, write manifest
    uv run python -m scripts.fetch_corpus --verify  # check local files against the manifest
"""

import argparse
import hashlib
import json
import posixpath
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

FASTAPI_VERSION = "0.143.0"
FASTAPI_URL = f"https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/{FASTAPI_VERSION}"
FASTAPI_SECTION = "docs/en/docs/tutorial/"
# Include paths like "../../docs_src/x.py" are relative to the MkDocs project directory.
FASTAPI_INCLUDE_BASE = "docs/en"

PYTHON_DOCS_VERSION = "3.12"
PYTHON_DOCS_URL = (
    f"https://docs.python.org/{PYTHON_DOCS_VERSION}/archives/"
    f"python-{PYTHON_DOCS_VERSION}-docs-pdf-a4.zip"
)
PYTHON_HOWTOS = ["howto-argparse.pdf", "howto-logging.pdf", "howto-regex.pdf", "howto-sorting.pdf"]

# Research papers (Part 4b). Only papers whose arXiv license allows redistribution (CC BY or
# CC BY-SA, checked on each abstract page); most arXiv papers use a licence that does not.
# Versions are pinned so the text never changes under the benchmark.
ARXIV_PAPERS = [
    {"id": "2309.15217", "version": 2, "name": "ragas", "license": "CC-BY-4.0",
     "title": "Ragas: Automated Evaluation of Retrieval Augmented Generation"},
    {"id": "2311.09476", "version": 2, "name": "ares", "license": "CC-BY-4.0",
     "title": "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems"},
    {"id": "2401.05856", "version": 1, "name": "rag-failure-points", "license": "CC-BY-4.0",
     "title": "Seven Failure Points When Engineering a Retrieval Augmented Generation System"},
    {"id": "2310.11511", "version": 1, "name": "self-rag", "license": "CC-BY-4.0",
     "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"},
    {"id": "1908.10084", "version": 1, "name": "sentence-bert", "license": "CC-BY-SA-4.0",
     "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"},
    {"id": "2212.03533", "version": 2, "name": "e5", "license": "CC-BY-4.0",
     "title": "Text Embeddings by Weakly-Supervised Contrastive Pre-training"},
    {"id": "2112.01488", "version": 3, "name": "colbertv2", "license": "CC-BY-4.0",
     "title": "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"},
    {"id": "2104.08663", "version": 4, "name": "beir", "license": "CC-BY-SA-4.0",
     "title": "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information "
              "Retrieval Models"},
]  # fmt: skip
# arXiv asks automated clients to wait between requests.
ARXIV_DELAY_SECONDS = 3

CORPUS_DIR = Path("evals/corpus")
RAW_DIR = CORPUS_DIR / "raw"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

_FENCE = re.compile(r"\s*(```|~~~)")
_INCLUDE = re.compile(r"\{\*\s*(\S+)(.*?)\*\}")
_LINE_RANGE = re.compile(r"ln\[([^\]]*)\]")
_HEADING_ANCHOR = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$")
# Three or more slashes: nested blocks such as "//// tab | Python 3.10+" use four.
_ADMONITION_OPEN = re.compile(r"/{3,}\s*(\w+)\s*(?:\|\s*(.+?))?\s*")
_ADMONITION_CLOSE = re.compile(r"/{3,}\s*")
_HTML_TAG = re.compile(r"</?(?:dfn|abbr|span|div|a|img|details|summary|br)\b[^>]*>")
_CODE_LANGUAGE = {".py": "python", ".toml": "toml", ".json": "json", ".sh": "bash"}


class CorpusError(Exception):
    pass


def normalize_fastapi_markdown(text: str, read_file: Callable[[str], str]) -> str:
    """Turn FastAPI's MkDocs dialect into plain Markdown, as a reader sees it on the website.

    - `{* ../../docs_src/x.py ln[1:5] hl[2] *}` includes become fenced code blocks
    - `## Heading { #anchor }` loses the anchor
    - `/// tip | Title` ... `///` admonitions become a bold "Tip: Title" line
    - presentational HTML tags are dropped, keeping their text
    Lines inside fenced code blocks are left untouched.
    """
    output: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            output.append(line)
        elif in_fence:
            output.append(line)
        elif include := _INCLUDE.fullmatch(line.strip()):
            output.extend(_render_include(include.group(1), include.group(2), read_file))
        elif opening := _ADMONITION_OPEN.fullmatch(line):
            kind, title = opening.group(1).title(), opening.group(2)
            output.append(f"**{kind}: {title}**" if title else f"**{kind}:**")
        elif _ADMONITION_CLOSE.fullmatch(line):
            continue
        else:
            if line.startswith("#"):
                line = _HEADING_ANCHOR.sub("", line)
            output.append(_HTML_TAG.sub("", line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip() + "\n"


def _render_include(relative_path: str, options: str, read_file: Callable[[str], str]) -> list[str]:
    path = posixpath.normpath(posixpath.join(FASTAPI_INCLUDE_BASE, relative_path))
    code = read_file(path)
    if line_range := _LINE_RANGE.search(options):
        code = _select_lines(code, line_range.group(1))
    language = _CODE_LANGUAGE.get(posixpath.splitext(path)[1], "")
    return [f"```{language}", *code.rstrip("\n").splitlines(), "```"]


def _select_lines(code: str, spec: str) -> str:
    """`spec` is 1-based and inclusive, e.g. "1:25" or "1:3,7" or "28:31"."""
    lines = code.splitlines()
    picked: list[str] = []
    for part in spec.split(","):
        start, _, end = part.strip().partition(":")
        picked.extend(lines[int(start) - 1 : int(end or start)])
    return "\n".join(picked)


def is_safe_member_path(path: str) -> bool:
    """Reject archive entries that try to escape the output folder ("zip slip")."""
    return (
        not path.startswith("/")
        and posixpath.normpath(path) == path
        and ".." not in path.split("/")
    )


def build(work_dir: Path) -> list[dict]:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    fastapi_archive = _download(FASTAPI_URL, work_dir / "fastapi.tar.gz")
    python_archive = _download(PYTHON_DOCS_URL, work_dir / "python-docs.zip")
    return [
        _build_fastapi(fastapi_archive),
        _build_python_howtos(python_archive),
        _build_arxiv_papers(work_dir),
    ]


def _build_fastapi(archive: Path) -> dict:
    prefix = f"fastapi-{FASTAPI_VERSION}/"
    documents = []
    # Members are read straight from the archive into memory; nothing is extracted as-is.
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name.removeprefix(prefix): m for m in tar.getmembers() if m.isfile()}

        def read_file(path: str) -> str:
            member = members.get(path)
            if member is None:
                raise CorpusError(f"Missing file in the FastAPI archive: {path}")
            return tar.extractfile(member).read().decode("utf-8")

        doc_paths = sorted(
            path for path in members if path.startswith(FASTAPI_SECTION) and path.endswith(".md")
        )
        for doc_path in doc_paths:
            if not is_safe_member_path(doc_path):
                raise CorpusError(f"Unsafe path in archive: {doc_path}")
            markdown = normalize_fastapi_markdown(read_file(doc_path), read_file)
            relative = "fastapi/" + doc_path.removeprefix("docs/en/docs/")
            documents.append(
                _write(relative, markdown.encode("utf-8"), source="fastapi", collection="docs")
            )

    return {
        "source": {
            "name": "fastapi",
            "url": FASTAPI_URL,
            "version": FASTAPI_VERSION,
            "license": "MIT",
            "archive_sha256": _sha256(archive.read_bytes()),
            "selection": f"{FASTAPI_SECTION}**/*.md",
            "preprocessing": (
                "MkDocs includes inlined as fenced code; heading anchors, admonition markers "
                "and presentational HTML tags removed (see normalize_fastapi_markdown)"
            ),
        },
        "documents": documents,
    }


def _build_python_howtos(archive: Path) -> dict:
    documents = []
    with zipfile.ZipFile(archive) as zf:
        for name in PYTHON_HOWTOS:
            data = zf.read(f"docs-pdf/{name}")
            documents.append(
                _write(f"python/{name}", data, source="python-howto", collection="docs")
            )
    return {
        "source": {
            "name": "python-howto",
            "url": PYTHON_DOCS_URL,
            "version": PYTHON_DOCS_VERSION,
            "license": "PSF-2.0",
            "archive_sha256": _sha256(archive.read_bytes()),
            "selection": ", ".join(PYTHON_HOWTOS),
            "preprocessing": "none (original PDFs)",
        },
        "documents": documents,
    }


def _build_arxiv_papers(work_dir: Path) -> dict:
    documents = []
    for n, paper in enumerate(ARXIV_PAPERS):
        if n:
            time.sleep(ARXIV_DELAY_SECONDS)
        versioned = f"{paper['id']}v{paper['version']}"
        pdf = _download(f"https://arxiv.org/pdf/{versioned}", work_dir / f"{paper['name']}.pdf")
        data = pdf.read_bytes()
        if not data.startswith(b"%PDF-"):
            raise CorpusError(f"arXiv did not return a PDF for {versioned}")
        entry = _write(f"papers/{paper['name']}.pdf", data, source="arxiv", collection="papers")
        entry |= {"arxiv": versioned, "title": paper["title"], "license": paper["license"]}
        documents.append(entry)
    return {
        "source": {
            "name": "arxiv",
            "url": "https://arxiv.org",
            "version": "pinned per paper",
            "license": "per document (CC BY 4.0 or CC BY-SA 4.0)",
            "selection": ", ".join(f"{p['id']}v{p['version']}" for p in ARXIV_PAPERS),
            "preprocessing": "none (original PDFs)",
        },
        "documents": documents,
    }


def _write(relative: str, data: bytes, *, source: str, collection: str) -> dict:
    path = RAW_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": relative,
        "source": source,
        "collection": collection,
        "sha256": _sha256(data),
        "bytes": len(data),
    }


def _download(url: str, destination: Path) -> Path:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as f:
        shutil.copyfileobj(response, f)
    return destination


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(groups: list[dict]) -> None:
    manifest = {
        "generated_by": "uv run python -m scripts.fetch_corpus",
        "sources": [group["source"] for group in groups],
        "documents": [doc for group in groups for doc in group["documents"]],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def verify() -> bool:
    manifest = json.loads(MANIFEST_PATH.read_text())
    problems = []
    for doc in manifest["documents"]:
        path = RAW_DIR / doc["path"]
        if not path.exists():
            problems.append(f"missing  {doc['path']}")
        elif _sha256(path.read_bytes()) != doc["sha256"]:
            problems.append(f"changed  {doc['path']}")
    for problem in problems:
        print(problem)
    print(f"{len(manifest['documents']) - len(problems)}/{len(manifest['documents'])} files match")
    return not problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="check files against the manifest")
    args = parser.parse_args()

    if args.verify:
        sys.exit(0 if verify() else 1)

    with tempfile.TemporaryDirectory() as work_dir:
        groups = build(Path(work_dir))
    write_manifest(groups)
    for group in groups:
        total = sum(doc["bytes"] for doc in group["documents"])
        name = group["source"]["name"]
        print(f"{name:14} {len(group['documents']):3} documents  {total / 1024:7.0f} KB")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
