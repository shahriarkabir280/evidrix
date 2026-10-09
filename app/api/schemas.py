"""Pydantic request and response models.

Component 1D · extended in 2G (chat) and 3D (retrieve).
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

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
