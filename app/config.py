from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "First-Principles RAG"
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    db_path: Path = Path("data/index.db")
    mistral_api_base: str = "https://api.mistral.ai/v1"
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
    embed_model: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1100"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
    retrieval_k: int = int(os.getenv("RAG_TOP_K", "8"))
    semantic_k: int = int(os.getenv("RAG_SEMANTIC_K", "12"))
    keyword_k: int = int(os.getenv("RAG_KEYWORD_K", "12"))
    min_evidence_score: float = float(os.getenv("RAG_MIN_EVIDENCE_SCORE", "0.18"))
    embed_batch_size: int = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))
    hyde_enabled: bool = os.getenv("RAG_HYDE_ENABLED", "true").lower() == "true"
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.75"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
