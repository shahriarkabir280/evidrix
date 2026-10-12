"""Pydantic request and response models.

Component 1D · extended in 2G (chat) and 3D (retrieve).
"""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.db.models import DocumentStatus, JobStatus
from app.rag.retrieval import RetrievalConfig


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    filename: str
    mime_type: str
    size_bytes: int
    content_hash: str
    status: DocumentStatus
    error_code: str | None
    error_message: str | None
    page_count: int | None
    created_at: datetime
    updated_at: datetime


class DocumentUploadOut(DocumentOut):
    # The job that will process the document; poll GET /jobs/{job_id}. None for a duplicate.
    job_id: uuid.UUID | None
    duplicate: bool


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    status: JobStatus
    attempts: int
    max_attempts: int
    last_error: str | None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class RetrievalOverride(BaseModel):
    """Per-request changes to the server's retrieval preset; omitted fields keep the preset."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["dense", "keyword", "hybrid"] | None = None
    rerank: bool | None = None
    top_k: int | None = None
    candidate_k: int | None = None


class QuestionRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    # Omit to search all of your completed documents.
    document_ids: Annotated[list[uuid.UUID], Field(min_length=1, max_length=100)] | None = None
    retrieval: RetrievalOverride | None = None


class ChatRequest(QuestionRequest):
    pass


class RetrieveRequest(QuestionRequest):
    pass


class SourceLocation(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    section_path: str | None
    page_start: int | None
    page_end: int | None


class CitationOut(SourceLocation):
    marker: int
    quote: str  # the start of the cited passage


class RetrievedOut(SourceLocation):
    rank: int
    score: float
    text: str
    # How each stage saw the chunk: dense/keyword rank and score, rrf_score, rerank_score, ...
    debug: dict[str, float]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int


class RetrievalConfigOut(BaseModel):
    retrieval: RetrievalConfig
    index_version: str
    embedding_model: str
    reranker_model: str | None


class ChatConfig(RetrievalConfigOut):
    llm_model: str
    prompt_version: str


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]
    usage: Usage
    timings_ms: dict[str, float]
    config: ChatConfig


class RetrieveResponse(BaseModel):
    hits: list[RetrievedOut]
    timings_ms: dict[str, float]
    config: RetrievalConfigOut
