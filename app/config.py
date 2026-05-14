from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)  # frozen = immutable after creation; safe to use as a module-level singleton
class Settings:
    app_name: str = "First-Principles RAG"
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    db_path: Path = Path("data/index.db")  # SQLite file; replaces an external vector DB

    mistral_api_base: str = "https://api.mistral.ai/v1"
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")  # empty fallback raises at call time, not at import
    chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")  # swap to mistral-large for better reasoning
    embed_model: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

    # 1100 chars ≈ 200-250 tokens — large enough for sentence context, small enough for a focused embedding
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1100"))
    # ~16% of chunk_size; ensures concepts that straddle a boundary appear fully in at least one chunk
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))

    retrieval_k: int = int(os.getenv("RAG_TOP_K", "8"))          # final chunks sent to the LLM after MMR
    semantic_k: int = int(os.getenv("RAG_SEMANTIC_K", "12"))     # wide funnel: candidates from cosine search before fusion
    keyword_k: int = int(os.getenv("RAG_KEYWORD_K", "12"))       # wide funnel: candidates from BM25 before fusion

    # below this fused score, the top-1 chunk is too weak — return "insufficient evidence" instead of generating
    min_evidence_score: float = float(os.getenv("RAG_MIN_EVIDENCE_SCORE", "0.18"))

    embed_batch_size: int = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))  # conservative batch size to avoid API rate limits

    # HyDE embeds a hypothetical answer passage to bridge the query/document embedding gap; costs one extra LLM call
    hyde_enabled: bool = os.getenv("RAG_HYDE_ENABLED", "true").lower() == "true"

    # 0.75 = 75% relevance, 25% diversity; prevents MMR from returning 8 chunks from the same paragraph
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.75"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
