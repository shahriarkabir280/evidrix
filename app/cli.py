"""Command-line tasks: ingest-dir, reindex, retitle, search, reap-jobs, and API users and keys.

Component 2D (ingest-dir, reindex, search) · 5B (reap-jobs) · 5C (create-user, create-key,
revoke-key).

Documents ingested here belong to the seeded dev user (dev@evidrix.local); to query them through
the API, create a key for that user with create-key.

    uv run python -m app.cli ingest-dir evals/corpus/raw
    uv run python -m app.cli reindex
    uv run python -m app.cli retitle
    uv run python -m app.cli search "How do I declare a cookie parameter?"
    uv run python -m app.cli search "OAuth2PasswordRequestForm" --preset dense -k 3
    uv run python -m app.cli reap-jobs
    uv run python -m app.cli create-user --email ada@example.com
    uv run python -m app.cli create-key --email dev@evidrix.local --name laptop
    uv run python -m app.cli revoke-key evx_AbCdEfGh
"""

import argparse
import time
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus, IndexVersion, User
from app.db.repositories import index_versions as index_versions_repo
from app.db.repositories import users as users_repo
from app.db.repositories.users import DEV_USER_ID
from app.db.session import get_session_factory
from app.rag.embeddings import get_embedder
from app.rag.rerank import get_reranker
from app.rag.retrieval import RetrievalConfig, load_preset, retrieve
from app.services.ingestion import (
    UploadRejected,
    ingest_upload,
    reap_stale_jobs,
    reindex_document,
    retitle_document,
)
from app.storage.blob import make_blob_store
from app.worker.queue import get_ingestion_queue

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def ingest_dir(directory: Path) -> None:
    settings = get_settings()
    embedder = get_embedder()
    blob_store = make_blob_store(settings)
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    counts = {"created": 0, "duplicate": 0, "failed": 0, "rejected": 0, "chunks": 0}
    started = time.perf_counter()

    for path in paths:
        name = path.relative_to(directory)
        with get_session_factory()() as session:
            try:
                result = ingest_upload(
                    session,
                    blob_store,
                    embedder,
                    owner_id=DEV_USER_ID,
                    filename=path.name,
                    data=path.read_bytes(),
                )
            except UploadRejected as exc:
                counts["rejected"] += 1
                print(f"  rejected   {name}: {exc.code}")
                continue
            document = result.document
            if result.job is None:  # already ingested (a failed one is processed again)
                counts["duplicate"] += 1
                print(f"  duplicate  {name}")
            elif document.status is DocumentStatus.FAILED:
                counts["failed"] += 1
                print(f"  failed     {name}: {document.error_code}")
            else:
                counts["created"] += 1
                counts["chunks"] += len(document.chunks)
                print(f"  ok         {name}  ({len(document.chunks)} chunks)")

    elapsed = time.perf_counter() - started
    print(
        f"\n{len(paths)} files in {elapsed:.1f}s: {counts['created']} ingested "
        f"({counts['chunks']} chunks), {counts['duplicate']} duplicates, "
        f"{counts['failed']} failed, {counts['rejected']} rejected"
    )


def reindex() -> None:
    """Build chunks for completed documents that have none in the active index version."""
    embedder = get_embedder()
    with get_session_factory()() as session:
        index_version = index_versions_repo.get_active(session)
        has_chunks = select(Chunk.document_id).where(Chunk.index_version_id == index_version.id)
        documents = session.scalars(
            select(Document).where(
                Document.status == DocumentStatus.COMPLETED, Document.id.not_in(has_chunks)
            )
        ).all()
        for document in documents:
            count = reindex_document(session, embedder, document)
            print(f"  reindexed  {document.title}  ({count} chunks)")
        print(f"{len(documents)} documents reindexed into {index_version.name}")


def retitle() -> None:
    """Re-read the titles of completed PDFs, for those ingested before display titles."""
    embedder = get_embedder()
    blob_store = make_blob_store(get_settings())
    with get_session_factory()() as session:
        documents = session.scalars(
            select(Document).where(
                Document.status == DocumentStatus.COMPLETED,
                Document.mime_type == "application/pdf",
            )
        ).all()
        changed = 0
        for document in documents:
            old_title = document.shown_title
            if retitle_document(session, embedder, blob_store, document):
                changed += 1
                print(f"  retitled   {old_title}  ->  {document.shown_title}")
        print(f"{changed} of {len(documents)} PDF titles changed")


def reap_jobs(lease_seconds: int) -> None:
    """Requeue (or fail) jobs left "processing" by a dead worker. Run before starting one."""
    with get_session_factory()() as session:
        result = reap_stale_jobs(session, timedelta(seconds=lease_seconds))
    queue = get_ingestion_queue()
    for job_id in result.requeued:
        queue.enqueue(job_id)
    print(
        f"{len(result.requeued)} stale jobs requeued, "
        f"{len(result.failed)} failed (no attempts left)"
    )


def create_user(email: str, key_name: str | None) -> None:
    with get_session_factory()() as session:
        user = User(email=email.strip().lower())
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            raise SystemExit(
                f"A user with email {email!r} already exists; use create-key."
            ) from None
        key = users_repo.create_api_key(session, user.id, key_name)
    _print_key(email, key)


def create_key(email: str, key_name: str | None) -> None:
    with get_session_factory()() as session:
        user = users_repo.get_by_email(session, email.strip().lower())
        if user is None:
            raise SystemExit(f"No user with email {email!r}; use create-user.")
        key = users_repo.create_api_key(session, user.id, key_name)
    _print_key(email, key)


def revoke_key(prefix: str) -> None:
    with get_session_factory()() as session:
        count = users_repo.revoke_api_keys(session, prefix)
    print(f"{count} key(s) revoked")


def _print_key(email: str, key: str) -> None:
    print(f"API key for {email} (shown once; only its hash is stored):\n\n  {key}\n")
    print(f'Use it as:  curl -H "Authorization: Bearer {key}" ...')


def index_version(action: str, name: str, chunk_size: int | None, overlap: int | None) -> None:
    with get_session_factory()() as session:
        if action == "create":
            if chunk_size is None:
                raise SystemExit("--chunk-size is required to create an index version.")
            overlap = overlap if overlap is not None else chunk_size // 6
            chunking = {
                "chunk_size_tokens": chunk_size,
                "max_tokens": round(chunk_size * 1.2),
                "overlap_tokens": overlap,
                "min_chunk_tokens": chunk_size // 4,
            }
            version = index_versions_repo.create_like_active(session, name, chunking)
            print(f"Created {version.name}: {version.chunking}. Activate it, then run reindex.")
        elif action == "activate":
            version = index_versions_repo.activate(session, name)
            print(f"{version.name} is now active.")
        else:
            for version in session.scalars(select(IndexVersion).order_by(IndexVersion.id)):
                mark = "*" if version.is_active else " "
                print(f"{mark} {version.name}  {version.chunking}")


def search(question: str, preset: str, k: int | None) -> None:
    config = load_preset(preset)
    if k is not None:
        config = RetrievalConfig(
            **{**config.model_dump(), "top_k": k, "candidate_k": max(k, config.candidate_k)}
        )
    with get_session_factory()() as session:
        index_version = index_versions_repo.get_active(session)
        result = retrieve(
            session,
            get_embedder(),
            get_reranker(),
            config,
            owner_id=DEV_USER_ID,
            index_version_id=index_version.id,
            question=question,
        )
    timings = ", ".join(f"{name} {ms:.0f} ms" for name, ms in result.timings_ms.items())
    print(f"{len(result.hits)} results ({preset}: {timings})\n")
    for hit in result.hits:
        where = hit.section_path or (f"page {hit.page_start}" if hit.page_start else "")
        if hit.page_start and hit.page_end != hit.page_start:
            where += f"-{hit.page_end}"
        trail = " ".join(
            f"{name}={value:g}" for name, value in hit.debug.items() if name.endswith("_rank")
        )
        snippet = " ".join(hit.text.split())[:160]
        print(f"{hit.rank}. [{hit.score:.3f}] {hit.document_title} · {where}  ({trail})")
        print(f"   {snippet}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-dir", help="ingest every supported file in a folder")
    ingest.add_argument("directory", type=Path)
    commands.add_parser("reindex", help="build missing chunks for the active index version")
    commands.add_parser("retitle", help="re-read PDF titles, e.g. after display titles were added")
    query = commands.add_parser("search", help="dense search over your documents")
    query.add_argument("question")
    query.add_argument(
        "--preset", default=get_settings().retrieval_preset, help="configs/retrieval/<name>.toml"
    )
    query.add_argument("-k", type=int, help="override the preset's top_k")
    reap = commands.add_parser("reap-jobs", help="recover jobs whose worker died mid-job")
    reap.add_argument(
        "--lease-seconds",
        type=int,
        default=get_settings().ingestion_lease_seconds,
        help="a job processing for longer than this has lost its worker",
    )
    for name, help_text in [
        ("create-user", "create an API user and print their first key"),
        ("create-key", "print a new key for an existing user"),
    ]:
        user_command = commands.add_parser(name, help=help_text)
        user_command.add_argument("--email", required=True)
        user_command.add_argument("--name", help="a label for the key, e.g. the machine")
    revoke = commands.add_parser("revoke-key", help="revoke keys by their prefix (evx_ + 8)")
    revoke.add_argument("prefix")
    versions = commands.add_parser(
        "index-version", help="list, create or activate index versions (chunking experiments)"
    )
    versions.add_argument("action", choices=["list", "create", "activate"])
    versions.add_argument("name", nargs="?", default="")
    versions.add_argument("--chunk-size", type=int, help="target tokens per chunk (create)")
    versions.add_argument("--overlap", type=int, help="overlap tokens (create; default size/6)")
    args = parser.parse_args()

    if args.command == "ingest-dir":
        ingest_dir(args.directory)
    elif args.command == "reindex":
        reindex()
    elif args.command == "retitle":
        retitle()
    elif args.command == "reap-jobs":
        reap_jobs(args.lease_seconds)
    elif args.command == "create-user":
        create_user(args.email, args.name)
    elif args.command == "create-key":
        create_key(args.email, args.name)
    elif args.command == "revoke-key":
        revoke_key(args.prefix)
    elif args.command == "index-version":
        index_version(args.action, args.name, args.chunk_size, args.overlap)
    else:
        search(args.question, args.preset, args.k)


if __name__ == "__main__":
    main()
