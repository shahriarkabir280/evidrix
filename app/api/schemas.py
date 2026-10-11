"""Pydantic request and response models.

Component 1D · extended in 2G (chat) and 3D (retrieve).
"""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.db.models import DocumentStatus


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
    duplicate: bool


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class ChatRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    # Omit to search all of your completed documents.
    document_ids: Annotated[list[uuid.UUID], Field(min_length=1, max_length=100)] | None = None


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


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int


class ChatConfig(BaseModel):
    top_k: int
    index_version: str
    embedding_model: str
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
