# RAG Pipeline — Dataclass & Transformation Flow

## Ingestion Path (`POST /ingest`)

```
PDF file on disk
        │
        │  extract_pdf_pages()          pdf_utils.py
        ▼
┌─────────────────────────────┐
│ ExtractedPage               │  one per PDF page
│   page_number: int          │
│   text: str                 │
│   section_title: str | None │
└─────────────────────────────┘
        │
        │  looks_like_references_page() pdf_utils.py
        │  called in rag_service.py
        │  → drops bibliography pages
        │
        │  chunk_pages()                chunking.py
        ▼
┌─────────────────────────────┐
│ Chunk                       │  one per text window
│   text: str                 │
│   page_start: int           │
│   page_end: int             │
│   section_title: str | None │
└─────────────────────────────┘
        │
        │  mistral_client.embed_texts() mistral_client.py
        │  → list[list[float]]
        │
        │  normalize()                  retrieval.py
        │  → normalized_text: str
        │
        │  storage.insert_document()    storage.py
        │  storage.insert_chunks()      storage.py
        │  → written to SQLite as raw rows
        │
        │  rag_service.ingest_files()   rag_service.py
        ▼
┌─────────────────────────────┐
│ IngestionFileResult         │  returned to caller
│   filename: str             │  (schemas.py)
│   document_id: int          │
│   chunks_created: int       │
└─────────────────────────────┘
```

---

## Query Path (`POST /query`)

```
User sends QueryRequest
┌─────────────────────────────┐
│ QueryRequest                │  schemas.py
│   query: str                │
│   top_k: int | None         │
└─────────────────────────────┘
        │
        │  detect_intent()              retrieval.py
        │  → "restricted" / "chitchat" → return early
        │  → "knowledge"               → continue
        │
        │  rewrite_query()             retrieval.py
        │  "how does attention work?"
        │  → "how does attention work? | keywords: attention, work"
        │
        │  mistral_client.embed_texts() mistral_client.py
        │  → query_embedding: list[float]
        │
        │  _hyde_embedding()            rag_service.py
        │  LLM generates hypothetical answer → embed it
        │  → hyde_embedding: list[float] | None
        │
        │  HybridRetriever._get_corpus() retrieval.py
        │  storage.fetch_chunks() → SQLite rows
        ▼
┌──────────────────────────────────────────┐
│ CorpusChunk                              │  one per chunk in DB
│   chunk_id, document_id, filename        │  built once, cached
│   text, normalized_text, section_title   │  in _corpus
│   page_start, page_end                   │
│   embedding: np.ndarray  ← json.loads()  │
│   tokens: tuple[str]     ← tokenize()    │
│   token_set: frozenset   ← frozenset()   │
└──────────────────────────────────────────┘
        │
        │  packed into RetrievalCorpus  retrieval.py
        ▼
┌──────────────────────────────────────────┐
│ RetrievalCorpus                          │  in-memory index
│   version: int            ← cache key    │
│   chunks: list[CorpusChunk]              │
│   by_id: dict[int, CorpusChunk]          │
│   term_frequencies: dict[int, Counter]   │
│   row_lengths: dict[int, int]            │
│   document_frequencies: Counter          │
│   avg_len: float                         │
└──────────────────────────────────────────┘
        │
        ├─ _semantic_search()  retrieval.py   → {chunk_id: cosine_score}
        ├─ _keyword_search()   retrieval.py   → {chunk_id: bm25_score}
        └─ _semantic_search()  retrieval.py   → {chunk_id: hyde_score}
                │
                │  _fuse()              retrieval.py
                │  normalize scores + RRF + boosts
                │  looks up CorpusChunk via corpus.by_id[chunk_id]
                ▼
        ┌─────────────────────────────────────┐
        │ RetrievalResult                     │  one per candidate chunk
        │   chunk_id, document_id, filename   │
        │   text, section_title               │
        │   page_start, page_end              │
        │   semantic_score: float             │
        │   keyword_score: float              │
        │   fused_score: float                │
        └─────────────────────────────────────┘
                │
                │  sort by fused_score
                │
                │  _deduplicate_results()  retrieval.py
                │  → drops 80%+ token-overlap duplicates
                │
                │  _mmr_select()           retrieval.py
                │  → picks final k diverse results
                │
                │  evidence gate           rag_service.py
                │  top score < 0.18 → return early
                │
                │  _build_messages()       rag_service.py
                │  → formats evidence for LLM prompt
                │
                │  mistral_client.chat()   mistral_client.py
                │  → answer: str
                │
                │  hallucination_check()   retrieval.py
                │  → rejects if >1/3 sentences unsupported
                │
                │  _citation_snippet()     rag_service.py
                │  _source_label()         rag_service.py
                ▼
        ┌─────────────────────────────────────┐
        │ Citation                            │  schemas.py
        │   label: str       ← "S1 p.3-4"    │  one per RetrievalResult
        │   document_id, filename, chunk_id   │
        │   section_title                     │
        │   page_start, page_end              │
        │   score: float                      │
        │   snippet: str                      │
        └─────────────────────────────────────┘
                │
                ▼
        ┌──────────────────────────────────────────┐
        │ QueryResponse                            │  schemas.py
        │   intent: "knowledge"                    │  returned to user
        │   rewritten_query: str                   │
        │   answer: str                            │
        │   citations: list[Citation]              │
        │   insufficient_evidence: bool            │
        │   debug: {top_score, semantic, keyword}  │
        └──────────────────────────────────────────┘
```

---

## Dataclass Summary

| Dataclass | Defined in | Created by | Used by |
|---|---|---|---|
| `ExtractedPage` | `pdf_utils.py` | `extract_pdf_pages()` | `chunk_pages()` |
| `Chunk` | `chunking.py` | `chunk_pages()` | `rag_service.ingest_files()` |
| `CorpusChunk` | `retrieval.py` | `_get_corpus()` | `_semantic_search()`, `_keyword_search()`, `_fuse()` |
| `RetrievalCorpus` | `retrieval.py` | `_get_corpus()` | `retrieve()` |
| `RetrievalResult` | `retrieval.py` | `_fuse()` | `_mmr_select()`, `rag_service.answer_query()` |
| `IngestionFileResult` | `schemas.py` | `rag_service.ingest_files()` | API response |
| `Citation` | `schemas.py` | `rag_service.answer_query()` | `QueryResponse` |
| `QueryRequest` | `schemas.py` | FastAPI (from request body) | `rag_service.answer_query()` |
| `QueryResponse` | `schemas.py` | `rag_service.answer_query()` | API response |