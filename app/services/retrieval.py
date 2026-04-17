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
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "please",
    "say",
    "solve",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
    "why",
}
SPELLING_NORMALIZATIONS = {
    "embeding": "embedding",
    "embeddingss": "embeddings",
    "tranformer": "transformer",
    "retrival": "retrieval",
    "attension": "attention",
    "genration": "generation",
}
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
    section_title: str | None
    page_start: int
    page_end: int
    semantic_score: float
    keyword_score: float
    fused_score: float


def normalize(text: str) -> str:
    return " ".join(_normalize_token(token) for token in WORD_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    return [_normalize_token(token) for token in WORD_RE.findall(text.lower())]


def _normalize_token(token: str) -> str:
    token = token.lower()
    return SPELLING_NORMALIZATIONS.get(token, token)


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
    important = [token for token in tokens if token not in STOP_WORDS]
    ordered_terms = list(dict.fromkeys(important))
    if not ordered_terms:
        return query
    return " | ".join(
        [
            query,
            "keywords: " + ", ".join(ordered_terms[:8]),
        ]
    )


class HybridRetriever:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def retrieve(
        self,
        rewritten_query: str,
        query_embedding: list[float],
        hyde_embedding: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        rows = self.storage.fetch_chunks()
        if not rows:
            return []
        limit = top_k or settings.retrieval_k
        semantic = self._semantic_search(rows, query_embedding, settings.semantic_k)
        keyword = self._keyword_search(rows, rewritten_query, settings.keyword_k)
        hyde_semantic = (
            self._semantic_search(rows, hyde_embedding, settings.semantic_k) if hyde_embedding else {}
        )
        merged = self._fuse(rows, rewritten_query, semantic, keyword, hyde_semantic)
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
        deduplicated = self._deduplicate_results(reranked, max(limit * 2, limit))
        return self._mmr_select(deduplicated, limit)

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

            score += self._technical_term_boost(query_terms, row["normalized_text"])
            if score > 0:
                scores.append((row_id, score))
        return dict(sorted(scores, key=lambda item: item[1], reverse=True)[:top_k])

    def _fuse(
        self,
        rows: list,
        query: str,
        semantic: dict[int, float],
        keyword: dict[int, float],
        hyde_semantic: dict[int, float],
    ) -> dict[int, RetrievalResult]:
        by_id = {row["id"]: row for row in rows}
        max_sem = max(semantic.values(), default=1.0)
        max_key = max(keyword.values(), default=1.0)
        max_hyde = max(hyde_semantic.values(), default=1.0)
        query_terms = set(tokenize(query))
        results: dict[int, RetrievalResult] = {}
        rrf_scores = self._rrf([semantic, keyword, hyde_semantic])
        for chunk_id in set(semantic) | set(keyword) | set(hyde_semantic):
            row = by_id[chunk_id]
            semantic_score = semantic.get(chunk_id, 0.0)
            keyword_score = keyword.get(chunk_id, 0.0)
            hyde_score = hyde_semantic.get(chunk_id, 0.0)
            normalized_sem = semantic_score / max_sem if max_sem else 0.0
            normalized_key = keyword_score / max_key if max_key else 0.0
            normalized_hyde = hyde_score / max_hyde if max_hyde else 0.0

            row_tokens = set(tokenize(row["normalized_text"]))
            overlap_tokens = row_tokens & query_terms
            coverage_boost = min(len(overlap_tokens) / 12.0, 0.15)
            exact_phrase_boost = self._phrase_boost(query, row["normalized_text"])
            heading_boost = self._heading_boost(query_terms, row["text"])
            section_boost = self._section_title_boost(query_terms, row["section_title"] or "")
            fused = (
                0.38 * normalized_sem
                + 0.26 * normalized_key
                + 0.24 * normalized_hyde
                + 2.0 * rrf_scores.get(chunk_id, 0.0)
                + coverage_boost
                + exact_phrase_boost
                + heading_boost
                + section_boost
            )
            results[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                section_title=row["section_title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                fused_score=fused,
            )
        return results

    def _rrf(self, rankings: list[dict[int, float]], k: int = 60) -> dict[int, float]:
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            ordered = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
            for rank, (chunk_id, _) in enumerate(ordered, start=1):
                fused[chunk_id] += 1.0 / (k + rank)
        return dict(fused)

    def _technical_term_boost(self, query_terms: list[str], normalized_text: str) -> float:
        tokens = tokenize(normalized_text)
        if not tokens:
            return 0.0

        dense_terms = [term for term in query_terms if len(term) >= 6]
        if not dense_terms:
            return 0.0

        matches = sum(1 for term in dense_terms if term in tokens)
        density = matches / max(1, len(dense_terms))
        return min(0.45, density * 0.45)

    def _phrase_boost(self, query: str, normalized_text: str) -> float:
        if "keywords:" not in query:
            return 0.0
        _, keyword_tail = query.split("keywords:", maxsplit=1)
        phrases = [item.strip() for item in keyword_tail.split(",") if len(item.strip()) >= 5]
        boost = 0.0
        for phrase in phrases:
            if phrase in normalized_text:
                boost += 0.04
        return min(0.16, boost)

    def _heading_boost(self, query_terms: set[str], original_text: str) -> float:
        lines = [line.strip().lower() for line in original_text.splitlines()[:3] if line.strip()]
        if not lines:
            return 0.0
        heading_tokens = set()
        for line in lines:
            heading_tokens.update(tokenize(line))
        overlap = len(query_terms & heading_tokens)
        return min(0.12, overlap * 0.04)

    def _section_title_boost(self, query_terms: set[str], section_title: str) -> float:
        if not section_title:
            return 0.0
        title_tokens = set(tokenize(section_title))
        overlap = len(query_terms & title_tokens)
        return min(0.14, overlap * 0.05)

    def _deduplicate_results(
        self,
        ranked: list[RetrievalResult],
        limit: int,
    ) -> list[RetrievalResult]:
        selected: list[RetrievalResult] = []
        seen_signatures: set[tuple[str, int]] = set()
        seen_prefixes: dict[tuple[str, int], list[set[str]]] = defaultdict(list)

        for item in ranked:
            normalized = normalize(item.text)
            signature = (item.filename, item.page_start)
            tokens = set(tokenize(normalized))
            if self._is_redundant_same_page(signature, tokens, seen_prefixes):
                continue
            if signature in seen_signatures:
                if item.page_end == item.page_start:
                    continue
            seen_signatures.add(signature)
            seen_prefixes[signature].append(tokens)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _is_redundant_same_page(
        self,
        signature: tuple[str, int],
        tokens: set[str],
        seen_prefixes: dict[tuple[str, int], list[set[str]]],
    ) -> bool:
        for seen in seen_prefixes.get(signature, []):
            overlap = len(tokens & seen)
            baseline = max(1, min(len(tokens), len(seen)))
            if overlap / baseline >= 0.8:
                return True
        return False

    def _mmr_select(self, ranked: list[RetrievalResult], limit: int) -> list[RetrievalResult]:
        if len(ranked) <= limit:
            return ranked

        selected: list[RetrievalResult] = []
        remaining = ranked[:]
        lambda_value = settings.mmr_lambda

        while remaining and len(selected) < limit:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                novelty_penalty = max(
                    self._chunk_similarity(candidate, chosen) for chosen in selected
                )
                mmr_score = lambda_value * candidate.fused_score - (1 - lambda_value) * novelty_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            selected.append(remaining.pop(best_index))

        return selected

    def _chunk_similarity(self, left: RetrievalResult, right: RetrievalResult) -> float:
        left_tokens = set(tokenize(left.text))
        right_tokens = set(tokenize(right.text))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


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
