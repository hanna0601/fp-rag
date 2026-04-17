from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestionFileResult(BaseModel):
    filename: str
    document_id: int
    chunks_created: int


class IngestionResponse(BaseModel):
    ingested: list[IngestionFileResult]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    label: str
    document_id: int
    filename: str
    chunk_id: int
    page_start: int
    page_end: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    intent: Literal["chitchat", "knowledge", "restricted"]
    rewritten_query: str | None = None
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    debug: dict[str, Any] = Field(default_factory=dict)
