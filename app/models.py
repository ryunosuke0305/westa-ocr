"""Pydantic models and data structures for the relay API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    RECEIVED = "RECEIVED"
    ENQUEUED = "ENQUEUED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    ERROR = "ERROR"


class Masters(BaseModel):
    ship_csv: str = Field(..., alias="shipCsv")
    item_csv: str = Field(..., alias="itemCsv")

    class Config:
        populate_by_name = True


class WebhookConfig(BaseModel):
    url: HttpUrl
    token: str = Field(..., min_length=1)


class GeminiConfig(BaseModel):
    model: Optional[str] = Field(default=None, description="Gemini model name")
    temperature: Optional[float] = None
    topP: Optional[float] = Field(default=None, alias="topP")
    topK: Optional[int] = Field(default=None, alias="topK")
    maxOutputTokens: Optional[int] = Field(default=None, alias="maxOutputTokens")

    class Config:
        populate_by_name = True


class JobOptions(BaseModel):
    split_mode: str = Field(default="pdf", alias="splitMode")
    dpi: Optional[int] = None
    concurrency: Optional[int] = None

    class Config:
        populate_by_name = True


class JobRequest(BaseModel):
    order_id: str = Field(..., alias="orderId")
    file_id: str = Field(..., alias="fileId")
    prompt: str
    pattern: Optional[str] = None
    masters: Masters
    webhook: WebhookConfig
    gemini: Optional[GeminiConfig] = None
    options: Optional[JobOptions] = None
    idempotency_key: Optional[str] = Field(default=None, alias="idempotencyKey")

    class Config:
        populate_by_name = True


class JobResponse(BaseModel):
    job_id: str = Field(..., alias="job_id")
    correlation_id: str = Field(..., alias="correlation_id")
    status: JobStatus


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class PageMeta(BaseModel):
    model: Optional[str] = None
    durationMs: Optional[int] = None
    tokensInput: Optional[int] = None
    tokensOutput: Optional[int] = None


class PageResult(BaseModel):
    pageIndex: int
    status: str
    isNonOrderPage: bool = False
    rawText: Optional[str] = None
    error: Optional[str] = None
    meta: Optional[PageMeta] = None


class JobDetail(BaseModel):
    jobId: str
    orderId: str
    status: JobStatus
    fileId: str
    prompt: str
    pattern: Optional[str]
    masters: Masters
    webhookUrl: HttpUrl
    webhookToken: str
    createdAt: datetime
    updatedAt: datetime
    totalPages: Optional[int]
    processedPages: Optional[int]
    skippedPages: Optional[int]
    lastError: Optional[str]
    pages: List[PageResult] = Field(default_factory=list)


__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "GeminiConfig",
    "JobDetail",
    "JobOptions",
    "JobRequest",
    "JobResponse",
    "JobStatus",
    "Masters",
    "PageMeta",
    "PageResult",
    "WebhookConfig",
]
