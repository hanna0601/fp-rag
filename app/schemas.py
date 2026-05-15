from __future__ import annotations

# Any — used for the debug dict whose values can be str, float, int, etc.
# Literal — restricts a field to a fixed set of string values (like an enum but lighter)
from typing import Any, Literal

# BaseModel — Pydantic's base class; handles parsing, validation, and JSON serialization
# Field — lets us attach metadata (constraints, defaults, descriptions) to a model field
from pydantic import BaseModel, Field

# Why Pydantic instead of plain dataclasses?
# FastAPI uses Pydantic to validate incoming request bodies and serialize outgoing responses.
# If a required field is missing or a value violates a constraint (e.g. query is empty),
# Pydantic raises a ValidationError that FastAPI converts to a 422 Unprocessable Entity response
# with a clear error message — no manual validation needed in route handlers.


# --- Ingestion schemas (used by POST /ingest) ---

class IngestionFileResult(BaseModel):
    """Result for a single file in an ingestion request.

    One of these is created per uploaded PDF and bundled into IngestionResponse.
    document_id is the auto-generated SQLite primary key — the client can use it
    to identify which document's chunks to inspect or delete later.
    """
    filename: str       # original upload filename, e.g. "attention.pdf"
    document_id: int    # auto-assigned SQLite id — unique per document, used in citations
    chunks_created: int # how many text chunks were extracted and indexed; 0 would indicate a parsing failure


class IngestionResponse(BaseModel):
    """Top-level response returned by POST /ingest.

    Wraps a list so the client can upload multiple PDFs in one multipart request
    and receive one result per file. The list is always in the same order as the uploads.
    """
    ingested: list[IngestionFileResult]


# --- Query schemas (used by POST /query) ---

class QueryRequest(BaseModel):
    """Parsed and validated body of a POST /query request.

    FastAPI reads the JSON body, passes it to Pydantic, and only calls the route
    handler if validation succeeds. If query is empty or top_k is out of range,
    FastAPI returns 422 Unprocessable Entity automatically.
    """
    # min_length=1 prevents empty-string queries — the retrieval pipeline cannot
    # handle them and would waste an embedding API call on a zero-length input
    query: str = Field(min_length=1)

    # None means "use the default from settings.retrieval_k"
    # ge=1 (greater-or-equal) and le=20 (less-or-equal) enforce 1 ≤ top_k ≤ 20
    # int | None is equivalent to Optional[int] — the field is optional in the JSON body
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    """One cited source chunk returned alongside the answer.

    Each citation links the answer back to a specific passage in a specific PDF,
    enabling the user to verify claims. The UI uses these to render a source panel.

    label       — display string shown in the answer text, e.g. "[S1 p.3]" or "[S2 p.5-6]"
    document_id — SQLite id of the parent document (links to IngestionFileResult.document_id)
    chunk_id    — SQLite id of the specific chunk (for deep linking or debugging)
    score       — fused retrieval score (0-1 range after fusion); higher = more relevant
    snippet     — the 2 most query-relevant sentences from the chunk, shown as a preview
    """
    label: str
    document_id: int
    filename: str                  # e.g. "attention.pdf" — shown as the source file name in the UI
    chunk_id: int
    section_title: str | None = None  # e.g. "3. Attention Mechanism" — None if no heading was detected
    page_start: int                # first PDF page covered by this chunk
    page_end: int                  # last PDF page covered; page_start == page_end for single-page chunks
    score: float                   # rounded to 4 decimal places in rag_service; used to sort citations
    snippet: str                   # 2-sentence preview extracted by _citation_snippet()


class QueryResponse(BaseModel):
    """Full response returned by POST /query.

    The intent field determines which other fields are populated:

      "chitchat"   — answer is a fixed greeting; no citations, no rewritten_query
      "restricted" — answer is a refusal message; insufficient_evidence=True; no citations
      "knowledge"  — full pipeline ran; answer is LLM-generated; citations list is populated
                     unless insufficient_evidence=True (top chunk score too low or hallucination detected)

    insufficient_evidence=True means the system refused to generate an answer because
    the retrieved evidence was too weak. The client should show a "no relevant content" message
    rather than the empty answer string.

    debug is a free-form dict for development visibility:
      - "top_score": fused score of the best retrieved chunk
      - "semantic_score": cosine similarity of the top chunk
      - "keyword_score": BM25 score of the top chunk
      - "threshold": the min_evidence_score threshold (included when evidence was insufficient)
    """
    # Literal restricts the value to exactly one of these three strings at the type-system level
    # — Pydantic raises ValidationError if any other string is returned from the service layer
    intent: Literal["chitchat", "knowledge", "restricted"]

    # None for chitchat/restricted — only set for knowledge queries that went through rewrite_query()
    rewritten_query: str | None = None

    answer: str  # the LLM-generated answer, a fixed reply, or "insufficient evidence"

    # default_factory=list means each response gets its OWN empty list, not a shared one
    # (a common Python gotcha: default=[] shares the same list object across all instances)
    citations: list[Citation] = Field(default_factory=list)

    # True when the system chose not to generate an answer (evidence gate or hallucination check)
    insufficient_evidence: bool = False

    # Free-form diagnostic data for debugging retrieval quality — not shown to end users
    debug: dict[str, Any] = Field(default_factory=dict)
