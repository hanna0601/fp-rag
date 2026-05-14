from __future__ import annotations

import json                              # deserialize embedding JSON string → list[float]
import math                              # math.log used in BM25 IDF formula
import re                                # regex for tokenization and intent pattern matching
from collections import Counter, defaultdict  # Counter: term frequency counts; defaultdict: avoids KeyError on missing keys
from dataclasses import dataclass        # clean data containers with no boilerplate

import numpy as np                       # fast vector math for cosine similarity

from app.config import settings          # chunk sizes, k values, thresholds, lambda
from app.services.storage import Storage # fetch all chunks from SQLite


# --- Module-level constants (compiled once at import time, reused on every query) ---

# matches any word of 2+ alphanumeric characters — filters out single chars and punctuation
# e.g. "Hello, world!" → ["Hello", "world"]
WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")

# words that alone signal a greeting — used in detect_intent to catch "hi thanks bye"
GREETING_WORDS = {"hello", "hi", "hey", "thanks", "thank", "goodbye", "bye"}

# common words with no retrieval value — removed during query rewriting
# keeping them wastes BM25 budget on terms that appear in almost every chunk
# note: domain-specific words like "paper", "tell", "please" are included
# because users often say "tell me about..." or "what does this paper say..."
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "does", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "paper", "please", "say", "solve",
    "tell", "that", "the", "this", "to", "what", "which", "who", "why",
}

# hand-built corrections for common misspellings in this domain
# applied during tokenization so BM25 matches "attension" against chunks containing "attention"
SPELLING_NORMALIZATIONS = {
    "embeding": "embedding",
    "embeddingss": "embeddings",
    "tranformer": "transformer",
    "retrival": "retrieval",
    "attension": "attention",
    "genration": "generation",
}

# queries matching any of these patterns are refused immediately — no retrieval, no LLM call
# re.I = case-insensitive so "SSN", "ssn", "Ssn" all match
# \b = word boundary so "passport" matches but "passporting" does not
RESTRICTED_PATTERNS = {
    "pii": re.compile(r"\b(ssn|social security|credit card|passport|driver'?s license)\b", re.I),
    "medical": re.compile(r"\b(diagnose|prescribe|treatment plan|medical advice)\b", re.I),
    "legal": re.compile(r"\b(legal advice|sue|lawsuit|contract advice)\b", re.I),
}

# queries matching any of these are answered with a fixed greeting — no retrieval needed
# ^ = start of string, \b = word boundary, re.I = case-insensitive
CHITCHAT_PATTERNS = [
    re.compile(r"^\s*(hello|hi|hey)\b", re.I),   # starts with a greeting word
    re.compile(r"\bwho are you\b", re.I),
    re.compile(r"\bwhat can you do\b", re.I),
    re.compile(r"\bhow are you\b", re.I),
    re.compile(r"\bthank(s| you)?\b", re.I),      # "thanks", "thank you", "thank"
]


# --- Data classes (plain containers, no logic) ---

@dataclass
class RetrievalResult:
    """One retrieved chunk with all scores attached — returned to rag_service.py."""
    chunk_id: int
    document_id: int
    filename: str             # e.g. "attention.pdf" — used in citation labels
    chunk_index: int
    text: str                 # raw prose shown to the LLM as evidence
    section_title: str | None
    page_start: int
    page_end: int
    semantic_score: float     # raw cosine similarity score (before normalization)
    keyword_score: float      # raw BM25 score (before normalization)
    fused_score: float        # final combined score used for ranking


@dataclass
class CorpusChunk:
    """One chunk as stored in the in-memory corpus — richer than RetrievalResult.

    Carries pre-computed fields (tokens, token_set, embedding as np.ndarray)
    so that BM25 and cosine similarity don't re-compute them on every query.
    """
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    text: str
    normalized_text: str      # lowercased, cleaned — used as input to tokenize()
    section_title: str | None
    page_start: int
    page_end: int
    embedding: np.ndarray     # deserialized from JSON string; dtype=float for numpy math
    tokens: tuple[str, ...]   # tokenized form of normalized_text — tuple for hashability
    token_set: frozenset[str] # unique tokens — used for fast set intersection in dedup and boosts


@dataclass
class RetrievalCorpus:
    """The entire knowledge base loaded into memory for fast in-process search.

    Rebuilt from SQLite only when storage.cache_version changes (new upload or reset).
    Cached as self._corpus in HybridRetriever between queries.
    """
    version: int                           # matches storage.cache_version at build time
    chunks: list[CorpusChunk]             # all chunks in insertion order
    by_id: dict[int, CorpusChunk]         # chunk_id → chunk for O(1) lookup during fusion
    term_frequencies: dict[int, Counter]  # chunk_id → {term: count} — used in BM25 TF
    row_lengths: dict[int, int]           # chunk_id → token count — used in BM25 length norm
    document_frequencies: Counter         # term → how many chunks contain it — used in BM25 IDF
    avg_len: float                        # average token count across all chunks — BM25 length norm baseline


# --- Standalone utility functions ---

def normalize(text: str) -> str:
    """Return a cleaned string of tokens joined by spaces.

    Used for citation snippet generation and deduplication comparison.
    Example: "Hello, World! 123" → "hello world 123"
    """
    return " ".join(_normalize_token(token) for token in WORD_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    """Return a list of normalized tokens from text.

    Used everywhere: BM25 scoring, query rewriting, intent detection, MMR similarity.
    Example: "Attention Mechanism" → ["attention", "mechanism"]
             "attension" → ["attention"]  (spelling fix applied)
    """
    return [_normalize_token(token) for token in WORD_RE.findall(text.lower())]


def _normalize_token(token: str) -> str:
    token = token.lower()
    return SPELLING_NORMALIZATIONS.get(token, token)  # fix typo if known, else return as-is


def detect_intent(query: str) -> str:
    """Classify the query as 'restricted', 'chitchat', or 'knowledge'.

    Checked in order of priority:
      1. restricted — matches PII / medical / legal patterns → refuse immediately
      2. chitchat   — matches greeting patterns or is pure greeting words → fixed reply
      3. knowledge  — everything else → proceed to retrieval

    Why regex and not an LLM?
    An LLM call for intent adds ~500ms latency and API cost on every query.
    Regex is 0ms and free. Patterns cover the vast majority of non-knowledge queries.
    """
    stripped = query.strip().lower()
    if not stripped:                                                      # empty query → chitchat
        return "chitchat"
    if any(pattern.search(query) for pattern in RESTRICTED_PATTERNS.values()):  # PII/medical/legal
        return "restricted"
    if any(pattern.search(query) for pattern in CHITCHAT_PATTERNS):      # greeting patterns
        return "chitchat"
    tokens = tokenize(stripped)
    if not tokens:                                                        # only punctuation/numbers
        return "chitchat"
    if len(tokens) <= 4 and all(token in GREETING_WORDS for token in tokens):  # "hi thanks bye"
        return "chitchat"
    return "knowledge"                                                    # anything substantive


def rewrite_query(query: str) -> str:
    """Append a keyword suffix to the query to boost BM25 matching.

    Output format: "original query | keywords: term1, term2, ..."

    Why? BM25 scores exact term matches. Adding an explicit keyword list gives BM25
    a cleaner signal without stop words diluting the score.
    The original query is kept so the LLM and semantic search see natural language.

    Example:
      input:  "How does the attention mechanism work in transformers?"
      output: "How does the attention mechanism work in transformers? | keywords: attention, mechanism, work, transformers"
    """
    query = query.strip()
    tokens = tokenize(query)                                    # tokenize the full query
    if not tokens:
        return query
    important = [token for token in tokens if token not in STOP_WORDS]  # drop "how", "does", "the", "in"
    ordered_terms = list(dict.fromkeys(important))              # deduplicate while preserving order
                                                                # dict.fromkeys keeps first occurrence only
    if not ordered_terms:
        return query
    return " | ".join(
        [
            query,                                              # original natural language query
            "keywords: " + ", ".join(ordered_terms[:8]),       # top 8 important terms as explicit signal
        ]
    )


class HybridRetriever:
    """Combines semantic search, BM25 keyword search, and HyDE into a single ranked list.

    Holds the corpus in memory between queries (_corpus) and only rebuilds it
    when new documents are ingested (detected via storage.cache_version).
    """

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._corpus: RetrievalCorpus | None = None  # None until first query

    def retrieve(
        self,
        rewritten_query: str,
        query_embedding: list[float],
        hyde_embedding: list[float] | None = None,  # None if HyDE is disabled or failed
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Run the full hybrid retrieval pipeline and return top-k diverse results."""
        corpus = self._get_corpus()        # load from SQLite or return cached version
        if not corpus.chunks:              # no documents ingested yet
            return []
        limit = top_k or settings.retrieval_k  # use caller's top_k or fall back to config default

        # wide funnel: each search returns more candidates than the final limit
        semantic = self._semantic_search(corpus, query_embedding, settings.semantic_k)    # cosine similarity
        keyword = self._keyword_search(corpus, rewritten_query, settings.keyword_k)      # BM25
        hyde_semantic = (
            self._semantic_search(corpus, hyde_embedding, settings.semantic_k) if hyde_embedding else {}
            # HyDE: embed a hypothetical answer and search with that — bridges query/doc embedding gap
            # empty dict if HyDE is disabled so _fuse() treats it as contributing nothing
        )

        merged = self._fuse(corpus, rewritten_query, semantic, keyword, hyde_semantic)
        # sort by fused_score descending; ties broken by semantic, then keyword, then chunk length
        reranked = sorted(
            merged.values(),
            key=lambda item: (
                item.fused_score,
                item.semantic_score,
                item.keyword_score,
                len(tokenize(item.text)),  # longer chunks as last tiebreaker (more content = preferred)
            ),
            reverse=True,
        )
        # deduplicate before MMR — remove near-identical chunks from the same page
        # limit * 2 gives MMR a larger pool to pick from for better diversity
        deduplicated = self._deduplicate_results(reranked, max(limit * 2, limit))
        return self._mmr_select(deduplicated, limit)  # narrow funnel: pick final k diverse results

    def _get_corpus(self) -> RetrievalCorpus:
        """Return the in-memory corpus, rebuilding from SQLite only if data has changed.

        cache_version acts as a dirty flag: incremented by Storage on every write/reset.
        If versions match, the in-memory corpus is still valid — skip the DB round trip.
        """
        if self._corpus and self._corpus.version == self.storage.cache_version:
            return self._corpus  # cache hit — no DB query needed

        # cache miss — fetch all chunks fresh and rebuild all BM25 data structures
        rows = self.storage.fetch_chunks()
        chunks: list[CorpusChunk] = []
        term_frequencies: dict[int, Counter] = {}   # {chunk_id: {term: count}}
        row_lengths: dict[int, int] = {}             # {chunk_id: token_count}
        document_frequencies = Counter()             # {term: how_many_chunks_contain_it}

        for row in rows:
            tokens = tuple(tokenize(row["normalized_text"]))  # pre-tokenize once; reused on every query
            token_freq = Counter(tokens)                       # {term: count} for this chunk
            chunk = CorpusChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                normalized_text=row["normalized_text"],
                section_title=row["section_title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                embedding=np.array(json.loads(row["embedding"]), dtype=float),  # JSON string → np.ndarray
                tokens=tokens,
                token_set=frozenset(token_freq),  # unique terms — used for fast set intersection
            )
            chunks.append(chunk)
            term_frequencies[chunk.chunk_id] = token_freq      # store for BM25 TF lookup
            row_lengths[chunk.chunk_id] = len(tokens)          # store for BM25 length normalization
            for term in token_freq:
                document_frequencies[term] += 1                # count how many chunks contain each term

        # avg_len = average tokens per chunk — BM25 uses this to penalize unusually long chunks
        # max(len(chunks), 1) avoids division by zero when corpus is empty
        avg_len = sum(row_lengths.values()) / max(len(chunks), 1) if chunks else 1.0
        self._corpus = RetrievalCorpus(
            version=self.storage.cache_version,                # snapshot the version at build time
            chunks=chunks,
            by_id={chunk.chunk_id: chunk for chunk in chunks}, # O(1) lookup by id during fusion
            term_frequencies=term_frequencies,
            row_lengths=row_lengths,
            document_frequencies=document_frequencies,
            avg_len=avg_len or 1.0,                            # fallback to 1.0 to avoid division by zero
        )
        return self._corpus

    def _semantic_search(
        self,
        corpus: RetrievalCorpus,
        query_embedding: list[float],
        top_k: int,
    ) -> dict[int, float]:
        """Score every chunk by cosine similarity to the query embedding.

        Cosine similarity = dot(query, chunk) / (||query|| * ||chunk||)
        Range: -1 to 1. Higher = more similar.

        Why cosine and not dot product?
        Dot product is sensitive to vector magnitude — a longer chunk could score
        higher just because it has more content, not because it's more relevant.
        Cosine normalizes for length so short and long chunks compete fairly.

        Returns: {chunk_id: cosine_score} for the top_k highest scoring chunks.
        """
        query = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query) or 1.0    # ||query||; fallback to 1.0 to avoid division by zero
        scores: list[tuple[int, float]] = []
        for chunk in corpus.chunks:
            denom = (np.linalg.norm(chunk.embedding) or 1.0) * query_norm  # ||chunk|| * ||query||
            score = float(np.dot(chunk.embedding, query) / denom)           # cosine similarity
            scores.append((chunk.chunk_id, score))
        # sort descending, keep only top_k — discard the rest before fusion
        return dict(sorted(scores, key=lambda item: item[1], reverse=True)[:top_k])

    def _keyword_search(self, corpus: RetrievalCorpus, query: str, top_k: int) -> dict[int, float]:
        """Score every chunk using BM25 (Best Match 25) keyword ranking.

        BM25 improves on TF-IDF with two additions:
          1. TF saturation (k1): a term appearing 10x is not 10x more relevant than 5x
          2. Length normalization (b): long chunks are penalized so a rare term in a
             short chunk outscores the same term buried in a very long chunk

        Formula per term t in query, for chunk d:
          IDF(t) * TF(t,d) * (k1+1) / (TF(t,d) + k1 * (1 - b + b * len(d)/avg_len))

        k1=1.5, b=0.75 are standard TREC-recommended defaults.

        Returns: {chunk_id: bm25_score} for chunks with score > 0.
        """
        query_terms = tokenize(query)   # includes the "keywords: ..." suffix terms from rewrite_query()
        if not query_terms:
            return {}
        doc_count = max(len(corpus.chunks), 1)  # total number of chunks — used in IDF

        k1 = 1.5   # TF saturation: controls how fast term frequency stops mattering
        b = 0.75   # length normalization strength: 0 = no normalization, 1 = full normalization
        scores: list[tuple[int, float]] = []
        for chunk in corpus.chunks:
            row_id = chunk.chunk_id
            doc_len = corpus.row_lengths[row_id] or 1   # token count for this chunk
            score = 0.0
            for term in query_terms:
                freq = corpus.term_frequencies[row_id][term]   # how many times term appears in this chunk
                if freq == 0:
                    continue   # term not in chunk — contributes nothing, skip to avoid log(0)

                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                # N = total chunks, df = chunks containing this term
                # rare terms (low df) → high IDF → high score
                # common terms (high df) → low IDF → low score
                idf = math.log(
                    1
                    + (doc_count - corpus.document_frequencies[term] + 0.5)
                    / (corpus.document_frequencies[term] + 0.5)
                )
                # TF component with saturation: freq * (k1+1) / (freq + k1 * length_norm_factor)
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / corpus.avg_len)
                score += idf * (numerator / denominator)

            score += self._technical_term_boost(query_terms, chunk.tokens)  # extra credit for long technical terms
            if score > 0:
                scores.append((row_id, score))
        return dict(sorted(scores, key=lambda item: item[1], reverse=True)[:top_k])

    def _fuse(
        self,
        corpus: RetrievalCorpus,
        query: str,
        semantic: dict[int, float],    # {chunk_id: cosine_score}
        keyword: dict[int, float],     # {chunk_id: bm25_score}
        hyde_semantic: dict[int, float], # {chunk_id: cosine_score} from HyDE embedding
    ) -> dict[int, RetrievalResult]:
        """Combine all three retrieval signals into a single fused score per chunk.

        Raw scores from different methods have different scales (cosine: -1 to 1,
        BM25: 0 to ~20). To combine them fairly, each is normalized to 0-1 by
        dividing by the maximum score in that ranklist.

        Final fused score formula:
          0.38 * norm_semantic
        + 0.26 * norm_keyword
        + 0.24 * norm_hyde
        + 2.0  * rrf_score        ← dominant signal; rank-based so scale-independent
        + coverage_boost          ← token overlap between query and chunk (up to 0.15)
        + exact_phrase_boost      ← keyword phrases found verbatim in chunk (up to 0.16)
        + heading_boost           ← query terms found in chunk's first 3 lines (up to 0.12)
        + section_boost           ← query terms found in section_title (up to 0.14)

        The union set(semantic) | set(keyword) | set(hyde_semantic) ensures every chunk
        that appeared in ANY of the three ranklists gets a fused score — not just those
        that appeared in all three.
        """
        # normalize each ranklist independently so scores are 0-1 before combining
        max_sem = max(semantic.values(), default=1.0)    # avoid division by zero if dict is empty
        max_key = max(keyword.values(), default=1.0)
        max_hyde = max(hyde_semantic.values(), default=1.0)
        query_terms = set(tokenize(query))
        results: dict[int, RetrievalResult] = {}
        rrf_scores = self._rrf([semantic, keyword, hyde_semantic])  # rank-based fusion scores

        # iterate over every chunk that appeared in at least one ranklist
        for chunk_id in set(semantic) | set(keyword) | set(hyde_semantic):
            chunk = corpus.by_id[chunk_id]                        # O(1) lookup
            semantic_score = semantic.get(chunk_id, 0.0)          # 0.0 if not in semantic results
            keyword_score = keyword.get(chunk_id, 0.0)
            hyde_score = hyde_semantic.get(chunk_id, 0.0)

            # normalize each score to 0-1 range for fair combination
            normalized_sem = semantic_score / max_sem if max_sem else 0.0
            normalized_key = keyword_score / max_key if max_key else 0.0
            normalized_hyde = hyde_score / max_hyde if max_hyde else 0.0

            # coverage_boost: reward chunks that share many tokens with the query
            # capped at 0.15 so it can't dominate the score
            overlap_tokens = chunk.token_set & query_terms
            coverage_boost = min(len(overlap_tokens) / 12.0, 0.15)

            exact_phrase_boost = self._phrase_boost(query, chunk.normalized_text)   # verbatim keyword match
            heading_boost = self._heading_boost(query_terms, chunk.text)             # query terms in first 3 lines
            section_boost = self._section_title_boost(query_terms, chunk.section_title or "")  # query terms in heading

            fused = (
                0.38 * normalized_sem    # semantic similarity — main signal
                + 0.26 * normalized_key  # keyword match — catches exact term queries
                + 0.24 * normalized_hyde # HyDE similarity — bridges query/doc style gap
                + 2.0 * rrf_scores.get(chunk_id, 0.0)  # RRF is the dominant signal (weight 2.0)
                + coverage_boost
                + exact_phrase_boost
                + heading_boost
                + section_boost
            )
            results[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                semantic_score=semantic_score,   # raw (un-normalized) — stored for debug output
                keyword_score=keyword_score,     # raw (un-normalized) — stored for debug output
                fused_score=fused,
            )
        return results

    def _rrf(self, rankings: list[dict[int, float]], k: int = 60) -> dict[int, float]:
        """Reciprocal Rank Fusion — combine multiple ranked lists by position, not score.

        For each chunk at rank r in a ranklist: score += 1 / (k + r)
        Summed across all ranklists.

        Why rank-based instead of score-based?
        BM25 scores and cosine scores have different scales and distributions.
        Normalizing them is imperfect. RRF ignores the actual score values entirely —
        it only cares about relative order (rank 1 is best, rank 2 is next, etc.).
        This makes it robust to scale differences between retrieval methods.

        k=60 is the standard default from the original RRF paper — it dampens the
        advantage of being ranked #1 vs #2 so the fusion is smooth.

        Example with k=60:
          rank 1 → 1/61 = 0.0164
          rank 2 → 1/62 = 0.0161
          rank 10 → 1/70 = 0.0143
        A chunk ranked #1 in all three lists scores ~3 * 0.0164 = 0.049
        """
        fused: dict[int, float] = defaultdict(float)  # defaultdict so missing keys start at 0.0
        for ranking in rankings:
            # sort each ranklist by score descending to get rank positions
            ordered = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
            for rank, (chunk_id, _) in enumerate(ordered, start=1):  # rank starts at 1, not 0
                fused[chunk_id] += 1.0 / (k + rank)  # add reciprocal rank contribution
        return dict(fused)

    def _technical_term_boost(self, query_terms: list[str], tokens: tuple[str, ...]) -> float:
        """Extra BM25 score for chunks that contain long technical terms from the query.

        Why? Short stop words like "is", "the" are already filtered. But medium-length
        technical terms like "attention", "transformer", "embedding" (6+ chars) are highly
        discriminative. If a chunk contains many of them, it's very likely relevant.

        density = matched_long_terms / total_long_terms_in_query
        boost = min(0.45, density * 0.45)  → max boost of 0.45 at 100% match
        """
        if not tokens:
            return 0.0
        dense_terms = [term for term in query_terms if len(term) >= 6]  # only long technical terms
        if not dense_terms:
            return 0.0
        matches = sum(1 for term in dense_terms if term in tokens)  # how many appear in this chunk
        density = matches / max(1, len(dense_terms))                # fraction matched
        return min(0.45, density * 0.45)

    def _phrase_boost(self, query: str, normalized_text: str) -> float:
        """Boost chunks where keyword phrases from the rewritten query appear verbatim.

        rewrite_query() appends "keywords: term1, term2, ..." to the query.
        This function extracts those terms and checks if any appear as substrings
        in the chunk's normalized text. Each match adds 0.04, capped at 0.16.

        Only triggers when the query has been rewritten (contains "keywords:").
        Only checks phrases of 5+ chars to avoid boosting short common words.
        """
        if "keywords:" not in query:   # unrewritten query — no keyword suffix to check
            return 0.0
        _, keyword_tail = query.split("keywords:", maxsplit=1)   # extract the "term1, term2, ..." part
        phrases = [item.strip() for item in keyword_tail.split(",") if len(item.strip()) >= 5]
        boost = 0.0
        for phrase in phrases:
            if phrase in normalized_text:  # substring match in the normalized chunk text
                boost += 0.04
        return min(0.16, boost)  # cap so this signal can't dominate the fused score

    def _heading_boost(self, query_terms: set[str], original_text: str) -> float:
        """Boost chunks whose first 3 lines contain query terms.

        The first few lines of a chunk are likely a section heading or topic sentence.
        If those lines share terms with the query, the chunk is probably about the right topic.
        Each overlapping term adds 0.04, capped at 0.12.
        """
        lines = [line.strip().lower() for line in original_text.splitlines()[:3] if line.strip()]
        if not lines:
            return 0.0
        heading_tokens = set()
        for line in lines:
            heading_tokens.update(tokenize(line))   # tokenize each heading line
        overlap = len(query_terms & heading_tokens)  # count shared terms
        return min(0.12, overlap * 0.04)

    def _section_title_boost(self, query_terms: set[str], section_title: str) -> float:
        """Boost chunks whose section_title contains query terms.

        section_title was detected by pdf_utils.extract_section_title() during ingestion.
        e.g. "3. Attention Mechanism" — if query mentions "attention", this chunk gets boosted.
        Each overlapping term adds 0.05, capped at 0.14.

        Slightly higher per-term weight than heading_boost (0.05 vs 0.04) because
        section_title is a more reliable signal — it was explicitly detected as a heading.
        """
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
        """Remove near-duplicate chunks before MMR selection.

        Why deduplicate before MMR?
        MMR penalizes similar chunks but still may include near-duplicates if they
        score high enough. This hard filter removes them first.

        Two chunks are considered duplicates if they share the same (filename, page_start)
        AND their token sets overlap by 80%+ (Jaccard-like overlap on the smaller set).

        signature = (filename, page_start) — same page, same file → candidate duplicate
        seen_prefixes — for each signature, track token sets of already-selected chunks
        """
        selected: list[RetrievalResult] = []
        seen_signatures: set[tuple[str, int]] = set()              # tracks (filename, page_start) pairs seen
        seen_prefixes: dict[tuple[str, int], list[set[str]]] = defaultdict(list)  # tokens per signature

        for item in ranked:
            normalized = normalize(item.text)
            signature = (item.filename, item.page_start)           # identity = file + page
            tokens = set(tokenize(normalized))
            if self._is_redundant_same_page(signature, tokens, seen_prefixes):
                continue   # 80%+ overlap with an already-selected chunk from the same page → skip
            if signature in seen_signatures:
                if item.page_end == item.page_start:               # single-page chunk already seen → skip
                    continue
            seen_signatures.add(signature)
            seen_prefixes[signature].append(tokens)                # record this chunk's tokens
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
        """Return True if this chunk overlaps 80%+ with any already-selected chunk from the same page.

        overlap / min(len(a), len(b)) >= 0.8 means: at least 80% of the smaller
        chunk's tokens appear in the larger chunk — they're essentially the same content.

        Example:
          chunk A tokens: {"attention", "mechanism", "query", "key", "value"}  (5 tokens)
          chunk B tokens: {"attention", "mechanism", "query", "key"}            (4 tokens)
          overlap = 4, baseline = min(5,4) = 4 → 4/4 = 1.0 → redundant
        """
        for seen in seen_prefixes.get(signature, []):
            overlap = len(tokens & seen)
            baseline = max(1, min(len(tokens), len(seen)))  # compare against the smaller set
            if overlap / baseline >= 0.8:
                return True
        return False

    def _mmr_select(self, ranked: list[RetrievalResult], limit: int) -> list[RetrievalResult]:
        """Maximal Marginal Relevance — pick k results that are relevant AND diverse.

        Greedy algorithm:
          1. Start with the highest-scoring chunk (most relevant).
          2. For each remaining candidate, compute:
               mmr_score = λ * fused_score - (1-λ) * max_similarity_to_selected
          3. Pick the candidate with the best MMR score.
          4. Repeat until k are selected.

        λ (lambda_value) = 0.75:
          75% weight on relevance, 25% penalty for similarity to already-selected chunks.
          Prevents all 8 results being from the same paragraph.

        _chunk_similarity uses Jaccard token overlap as a cheap proxy for semantic similarity.
        """
        if len(ranked) <= limit:
            return ranked   # already few enough — no need to filter

        selected: list[RetrievalResult] = []
        remaining = ranked[:]                 # copy so we can pop without mutating the input
        lambda_value = settings.mmr_lambda   # 0.75 = 75% relevance, 25% diversity

        while remaining and len(selected) < limit:
            if not selected:
                selected.append(remaining.pop(0))   # first pick is always the highest-scoring chunk
                continue

            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                # novelty_penalty = how similar this candidate is to the MOST similar already-selected chunk
                novelty_penalty = max(
                    self._chunk_similarity(candidate, chosen) for chosen in selected
                )
                mmr_score = lambda_value * candidate.fused_score - (1 - lambda_value) * novelty_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            selected.append(remaining.pop(best_index))  # add the best MMR candidate

        return selected

    def _chunk_similarity(self, left: RetrievalResult, right: RetrievalResult) -> float:
        """Jaccard token overlap between two chunks — used as similarity proxy in MMR.

        Jaccard = |intersection| / |union|
        Range: 0.0 (no shared tokens) to 1.0 (identical token sets)

        Example:
          left:  {"attention", "mechanism", "query"}
          right: {"attention", "mechanism", "value"}
          intersection = {"attention", "mechanism"} → 2
          union = {"attention", "mechanism", "query", "value"} → 4
          similarity = 2/4 = 0.5
        """
        left_tokens = set(tokenize(left.text))
        right_tokens = set(tokenize(right.text))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def hallucination_check(answer: str, evidence_chunks: list[RetrievalResult]) -> bool:
    """Return True if the answer appears to contain unsupported claims.

    Strategy: for each sentence in the answer, check whether ANY of its
    content words (>4 chars) appear in the evidence chunks. If no evidence
    term matches, the sentence is flagged as unsupported.

    If more than 1/3 of sentences are unsupported → hallucination detected → return True.

    Why >4 chars? Short words like "is", "the", "was" appear in every chunk and
    would make every sentence look "supported". Only meaningful content words matter.

    Limitation: this is lexical, not semantic. If the LLM paraphrases using synonyms
    ("summarize" instead of "condense"), the check may miss it. A stronger version
    would embed each sentence and compare cosine similarity to evidence chunks.

    Example:
      answer sentence: "The model uses residual connections."
      evidence terms contain: "residual", "connections" → supported ✓

      answer sentence: "The authors won a Nobel Prize."
      evidence terms contain: nothing matching "nobel", "prize" → unsupported ✗
    """
    # build a single Counter of all terms across all evidence chunks
    evidence_terms = Counter()
    for chunk in evidence_chunks:
        evidence_terms.update(tokenize(chunk.text))  # merge term counts from all chunks

    unsupported_sentences = 0
    # split answer into sentences at punctuation boundaries
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer) if segment.strip()]
    for sentence in sentences:
        terms = [term for term in tokenize(sentence) if len(term) > 4]  # content words only
        # if sentence has content words AND none appear in evidence → unsupported
        if terms and not any(evidence_terms[term] > 0 for term in terms):
            unsupported_sentences += 1

    # threshold: more than 1/3 of sentences unsupported → reject the whole answer
    # max(1, ...) ensures at least 1 unsupported sentence is needed even for very short answers
    return unsupported_sentences > max(1, len(sentences) // 3)
