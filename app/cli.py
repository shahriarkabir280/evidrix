"""Command-line tasks: ingest-dir, create-user, reindex.

Component 2D (ingest-dir, reindex, search) · 5C (create-user).

    uv run python -m app.cli ingest-dir evals/corpus/raw
    uv run python -m app.cli reindex
    uv run python -m app.cli search "How do I declare a cookie parameter?"
"""

import argparse
import time
from pathlib import Path

from sqlalchemy import select

from app.api.deps import DEV_USER_ID
from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.db.repositories import index_versions as index_versions_repo
from app.db.session import get_session_factory
from app.rag.embeddings import get_embedder
from app.rag.retrieval import dense_search
from app.services.ingestion import UploadRejected, ingest_upload, reindex_document
from app.storage.blob import LocalBlobStore

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def ingest_dir(directory: Path) -> None:
    settings = get_settings()
    embedder = get_embedder()
    blob_store = LocalBlobStore(settings.blob_dir)
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
            if not result.created:
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


def search(question: str, k: int) -> None:
    embedder = get_embedder()
    with get_session_factory()() as session:
        index_version = index_versions_repo.get_active(session)
        started = time.perf_counter()
        hits = dense_search(
            session,
            embedder,
            owner_id=DEV_USER_ID,
            index_version_id=index_version.id,
            question=question,
            k=k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"{len(hits)} results in {elapsed_ms:.0f} ms\n")
    for hit in hits:
        where = hit.section_path or (f"page {hit.page_start}" if hit.page_start else "")
        if hit.page_start and hit.page_end != hit.page_start:
            where += f"-{hit.page_end}"
        snippet = " ".join(hit.text.split())[:160]
        print(f"{hit.rank}. [{hit.score:.3f}] {hit.document_title} · {where}\n   {snippet}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-dir", help="ingest every supported file in a folder")
    ingest.add_argument("directory", type=Path)
    commands.add_parser("reindex", help="build missing chunks for the active index version")
    query = commands.add_parser("search", help="dense search over your documents")
    query.add_argument("question")
    query.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "ingest-dir":
        ingest_dir(args.directory)
    elif args.command == "reindex":
        reindex()
    else:
        search(args.question, args.k)


if __name__ == "__main__":
    main()
