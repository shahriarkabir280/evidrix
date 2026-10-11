"""Answer pipeline: retrieve, rerank, generate, persist.

Component 2G · rerank in 3C · persistence in 6A.
"""

import time
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import DocumentStatus
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import documents as documents_repo
from app.db.repositories import index_versions as index_versions_repo
from app.db.repositories.chunks import ChunkHit
from app.rag.embeddings import Embedder
from app.rag.generation import GeneratedAnswer, generate_answer
from app.rag.llm import LLMProvider
from app.services.ingestion import ensure_compatible


class DocumentNotFoundError(Exception):
    def __init__(self, document_id: uuid.UUID) -> None:
        super().__init__(f"Document {document_id} not found.")
        self.document_id = document_id


class DocumentNotReadyError(Exception):
    def __init__(self, document_id: uuid.UUID, status: DocumentStatus) -> None:
        super().__init__(f"Document {document_id} is {status.value}, not completed.")
        self.document_id = document_id
        self.status = status


@dataclass(frozen=True)
class ChatResult:
    generated: GeneratedAnswer
    hits: list[ChunkHit]
    timings_ms: dict[str, float]
    top_k: int
    index_version: str
    embedding_model: str
    llm_model: str


def answer_question(
    session: Session,
    embedder: Embedder,
    llm: LLMProvider,
    *,
    owner_id: uuid.UUID,
    question: str,
    document_ids: list[uuid.UUID] | None,
    top_k: int,
    max_tokens: int,
    min_score: float | None = None,
) -> ChatResult:
    started = time.perf_counter()
    if document_ids is not None:
        document_ids = list(dict.fromkeys(document_ids))  # drop duplicates, keep order
        check_documents(session, owner_id, document_ids)
    index_version = index_versions_repo.get_active(session)
    ensure_compatible(index_version, embedder)

    timings: dict[str, float] = {}
    with _timer(timings, "embed"):
        query_vector = embedder.embed_query(question)
    with _timer(timings, "retrieve"):
        hits = chunks_repo.search_dense(
            session,
            owner_id=owner_id,
            index_version_id=index_version.id,
            query_vector=query_vector,
            k=top_k,
            document_ids=document_ids,
        )
    with _timer(timings, "generate"):
        generated = generate_answer(llm, question, hits, max_tokens=max_tokens, min_score=min_score)
    timings["total"] = round((time.perf_counter() - started) * 1000, 1)

    return ChatResult(
        generated=generated,
        hits=hits,
        timings_ms=timings,
        top_k=top_k,
        index_version=index_version.name,
        embedding_model=embedder.model_name,
        llm_model=llm.model,
    )


def check_documents(session: Session, owner_id: uuid.UUID, document_ids: list[uuid.UUID]) -> None:
    """Every requested document must belong to the caller and be fully indexed.

    Someone else's document gets the same "not found" as a missing one, so the answer never
    reveals that the ID exists.
    """
    found = {d.id: d for d in documents_repo.get_many_for_owner(session, owner_id, document_ids)}
    for document_id in document_ids:
        document = found.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        if document.status is not DocumentStatus.COMPLETED:
            raise DocumentNotReadyError(document_id, document.status)


class _timer:
    """`with _timer(timings, "embed"):` records the block's duration in milliseconds."""

    def __init__(self, timings: dict[str, float], name: str) -> None:
        self._timings, self._name = timings, name

    def __enter__(self) -> None:
        self._started = time.perf_counter()

    def __exit__(self, *exc_info: object) -> None:
        self._timings[self._name] = round((time.perf_counter() - self._started) * 1000, 1)
