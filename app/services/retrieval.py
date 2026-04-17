from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.services.storage import Storage


WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")
GREETING_WORDS = {"hello", "hi", "hey", "thanks", "thank", "goodbye", "bye"}
RESTRICTED_PATTERNS = {
    "pii": re.compile(r"\b(ssn|social security|credit card|passport|driver'?s license)\b", re.I),
    "medical": re.compile(r"\b(diagnose|prescribe|treatment plan|medical advice)\b", re.I),
    "legal": re.compile(r"\b(legal advice|sue|lawsuit|contract advice)\b", re.I),
}


@dataclass
class RetrievalResult:
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    semantic_score: float
    keyword_score: float
    fused_score: float


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def detect_intent(query: str) -> str:
    stripped = query.strip().lower()
    if not stripped:
        return "chitchat"
    if any(pattern.search(query) for pattern in RESTRICTED_PATTERNS.values()):
        return "restricted"
    tokens = tokenize(stripped)
    if not tokens:
        return "chitchat"
    if len(tokens) <= 4 and all(token in GREETING_WORDS for token in tokens):
        return "chitchat"
    return "knowledge"


def rewrite_query(query: str) -> str:
    query = query.strip()
    tokens = tokenize(query)
    if not tokens:
        return query
    important = [token for token in tokens if token not in {"what", "is", "are", "the", "a", "an", "please"}]
    expansion = " ".join(dict.fromkeys(important))
    if not expansion:
        return query
    return f"{query}\nFocus terms: {expansion}"


class HybridRetriever:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def retrieve(self, rewritten_query: str, query_embedding: list[float], top_k: int | None = None) -> list[RetrievalResult]:
        rows = self.storage.fetch_chunks()
        if not rows:
            return []
        limit = top_k or settings.retrieval_k
        semantic = self._semantic_search(rows, query_embedding, settings.semantic_k)
        keyword = self._keyword_search(rows, rewritten_query, settings.keyword_k)
        merged = self._fuse(rows, rewritten_query, semantic, keyword)
        reranked = sorted(
            merged.values(),
            key=lambda item: (
                item.fused_score,
                item.semantic_score,
                item.keyword_score,
                len(tokenize(item.text)),
            ),
            reverse=True,
        )
        return reranked[:limit]

    def _semantic_search(self, rows: list, query_embedding: list[float], top_k: int) -> dict[int, float]:
        query = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query) or 1.0
        scores: list[tuple[int, float]] = []
        for row in rows:
            embedding = np.array(json.loads(row["embedding"]), dtype=float)
            denom = (np.linalg.norm(embedding) or 1.0) * query_norm
            score = float(np.dot(embedding, query) / denom)
            scores.append((row["id"], score))
        return dict(sorted(scores, key=lambda item: item[1], reverse=True)[:top_k])

    def _keyword_search(self, rows: list, query: str, top_k: int) -> dict[int, float]:
        query_terms = tokenize(query)
        if not query_terms:
            return {}
        doc_count = max(len(rows), 1)
        avg_len = sum(len(tokenize(row["normalized_text"])) for row in rows) / doc_count
        avg_len = avg_len or 1.0

        df = Counter()
        row_term_freq: dict[int, Counter] = {}
        row_lengths: dict[int, int] = {}
        for row in rows:
            terms = tokenize(row["normalized_text"])
            term_freq = Counter(terms)
            row_term_freq[row["id"]] = term_freq
            row_lengths[row["id"]] = len(terms)
            for term in term_freq:
                df[term] += 1

        k1 = 1.5
        b = 0.75
        scores: list[tuple[int, float]] = []
        for row in rows:
            row_id = row["id"]
            doc_len = row_lengths[row_id] or 1
            score = 0.0
            for term in query_terms:
                freq = row_term_freq[row_id][term]
                if freq == 0:
                    continue
                idf = math.log(1 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
                score += idf * (numerator / denominator)
            if score > 0:
                scores.append((row_id, score))
        return dict(sorted(scores, key=lambda item: item[1], reverse=True)[:top_k])

    def _fuse(
        self,
        rows: list,
        query: str,
        semantic: dict[int, float],
        keyword: dict[int, float],
    ) -> dict[int, RetrievalResult]:
        by_id = {row["id"]: row for row in rows}
        max_sem = max(semantic.values(), default=1.0)
        max_key = max(keyword.values(), default=1.0)
        query_terms = set(tokenize(query))
        results: dict[int, RetrievalResult] = {}
        for chunk_id in set(semantic) | set(keyword):
            row = by_id[chunk_id]
            semantic_score = semantic.get(chunk_id, 0.0)
            keyword_score = keyword.get(chunk_id, 0.0)
            normalized_sem = semantic_score / max_sem if max_sem else 0.0
            normalized_key = keyword_score / max_key if max_key else 0.0

            overlap_tokens = set(tokenize(row["normalized_text"])) & query_terms
            coverage_boost = min(len(overlap_tokens) / 12.0, 0.15)
            fused = 0.65 * normalized_sem + 0.35 * normalized_key + coverage_boost
            results[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                fused_score=fused,
            )
        return results


def hallucination_check(answer: str, evidence_chunks: list[RetrievalResult]) -> bool:
    evidence_terms = Counter()
    for chunk in evidence_chunks:
        evidence_terms.update(tokenize(chunk.text))

    unsupported_sentences = 0
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer) if segment.strip()]
    for sentence in sentences:
        terms = [term for term in tokenize(sentence) if len(term) > 4]
        if terms and not any(evidence_terms[term] > 0 for term in terms):
            unsupported_sentences += 1
    return unsupported_sentences > max(1, len(sentences) // 3)
