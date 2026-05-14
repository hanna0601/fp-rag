from __future__ import annotations

import json                          # serialize embeddings: list[float] → JSON string, e.g. [0.12, -0.34, ...]
import sqlite3                       # built-in Python database; no external vector DB needed
from collections.abc import Iterable # allows insert_chunks() to accept a generator or list
from pathlib import Path             # cross-platform file paths; safer than raw strings


class Storage:
    """SQLite-backed store for documents and their embedding chunks.

    Replaces an external vector database — embeddings are stored as JSON text
    in a regular SQLite column. The tradeoff: no ANN index, so retrieval does
    a full table scan (O(N) cosine similarity). Acceptable at small corpus size;
    swap to pgvector or FAISS for production scale.

    Schema:
        documents  — one row per uploaded PDF (filename, path, timestamp)
        chunks     — one row per text chunk, foreign-keyed to documents,
                     embedding stored as JSON float array

    Cache invalidation:
        _cache_version is an in-memory integer incremented on every write.
        HybridRetriever caches the full corpus in memory and compares its
        stored version to storage.cache_version on each query — if they match,
        it skips re-fetching from SQLite, avoiding redundant I/O.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path    # e.g. Path("data/index.db") — where the SQLite file lives on disk
        self._cache_version = 0   # starts at 0; goes up by 1 every time data is written or deleted
        self._initialize()        # create tables right away so the DB is always ready

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)  # open the file; creates it if it doesn't exist yet
        connection.row_factory = sqlite3.Row        # makes results dict-like: row["text"] instead of row[3]
                                                    # without this: row[3] — breaks silently if column order changes
        return connection

    def _initialize(self) -> None:
        """Create tables and index on first run; safe to call on every startup.

        executescript() runs multiple statements in one call — CREATE TABLE IF NOT EXISTS
        means re-running on an existing DB is a no-op (idempotent).

        The PRAGMA check at the end handles schema migrations: if the DB was created
        before section_title was added to the schema, ALTER TABLE adds it without
        losing existing data. This avoids a hard migration script.
        """
        with self.connect() as connection:   # 'with' = context manager: auto-commits on success, rolls back on error
            connection.executescript(
                # executescript runs the entire SQL block as one batch
                # IF NOT EXISTS means this is safe to call on every startup — it won't overwrite existing data
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- unique ID auto-assigned by SQLite on each insert
                    filename TEXT NOT NULL,                -- original name e.g. "attention.pdf" — shown in citations
                    stored_path TEXT NOT NULL,             -- timestamped path e.g. "data/uploads/20260417_attention.pdf"
                    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP  -- recorded automatically by SQLite
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,          -- which document this chunk came from (links to documents.id)
                    chunk_index INTEGER NOT NULL,          -- 0, 1, 2 ... — position of this chunk within its document
                    text TEXT NOT NULL,                    -- the actual prose text sent to the LLM as evidence
                    normalized_text TEXT NOT NULL,         -- lowercased, stop-words removed — used for BM25 scoring
                    section_title TEXT,                    -- e.g. "3. Attention Mechanism" — NULL if no heading found
                    page_start INTEGER NOT NULL,           -- first PDF page this chunk covers, e.g. 3
                    page_end INTEGER NOT NULL,             -- last PDF page, e.g. 4 — used for citation label "p.3-4"
                    embedding TEXT NOT NULL,               -- JSON float array e.g. "[0.12, -0.34, 0.98, ...]"
                                                           -- stored as TEXT because SQLite has no native float array type
                    FOREIGN KEY(document_id) REFERENCES documents(id)  -- enforces referential integrity
                );

                -- without this index, every JOIN in fetch_chunks() scans the whole chunks table
                -- with it, SQLite jumps directly to rows matching a given document_id
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                """
            )

            # PRAGMA table_info returns one row per column in the table
            # we collect just the "name" field from each row into a set for fast lookup
            # example result: {"id", "document_id", "text", "embedding", ...}
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
            }
            # migration guard: if this DB was created before section_title was added to the schema,
            # ALTER TABLE adds the column non-destructively — no data is lost, existing rows get NULL
            if "section_title" not in existing_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")

    def insert_document(self, filename: str, stored_path: str) -> int:
        """Insert a document record and return its auto-generated ID.

        The returned ID is used immediately by insert_chunks() to link all
        chunks of this document via the document_id foreign key.
        """
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO documents (filename, stored_path) VALUES (?, ?)",
                (filename, stored_path),  # ? placeholders — SQLite escapes the values, preventing SQL injection
                                          # never use f-strings here: f"... VALUES ('{filename}')" is vulnerable
            )
            return int(cursor.lastrowid)  # lastrowid = the auto-assigned id of the row just inserted
                                          # returned to rag_service.py so it can link chunks to this document

    def insert_chunks(self, rows: Iterable[dict]) -> None:
        """Bulk-insert all chunks for a document in a single transaction.

        Why materialize the generator into a list first?
        executemany() needs an iterable it can consume once. The generator from
        rag_service.py is lazy — materializing it here ensures all rows are ready
        before the DB transaction opens, so a partial failure doesn't leave the
        DB in a half-written state.

        Embeddings are JSON-serialized because SQLite has no native float array type.
        They are deserialized back to np.ndarray in HybridRetriever._get_corpus().

        _cache_version is incremented AFTER the write so the next retrieval call
        knows the in-memory corpus is stale and must be rebuilt from SQLite.
        """
        # materialize the generator into a list of tuples before opening the DB connection
        # if the generator raises mid-way, we fail before writing anything — no partial inserts
        payload = [
            (
                row["document_id"],           # foreign key — links this chunk to its parent document
                row["chunk_index"],           # 0-based position: first chunk = 0, second = 1, ...
                row["text"],                  # raw prose text used as evidence in the LLM prompt
                row["normalized_text"],       # cleaned version used by BM25 (lowercase, no stop words)
                row["section_title"],         # heading string or None — used for section-title retrieval boost
                row["page_start"],            # e.g. 3 — for citation label "p.3"
                row["page_end"],              # e.g. 4 — for citation label "p.3-4"
                json.dumps(row["embedding"]), # [0.12, -0.34, ...] → '[ 0.12, -0.34, ...]'
                                              # SQLite stores it as a string; deserialized back in _get_corpus()
            )
            for row in rows  # rows is a generator from rag_service.py — consumed exactly once here
        ]
        with self.connect() as connection:
            connection.executemany(   # inserts all rows in a single transaction — all succeed or all fail
                """
                INSERT INTO chunks (
                    document_id, chunk_index, text, normalized_text, section_title, page_start, page_end, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,              # list of tuples, one per chunk — matched positionally to the ? placeholders
            )
        self._cache_version += 1      # corpus changed — HybridRetriever will rebuild its in-memory index next query

    def fetch_chunks(self) -> list[sqlite3.Row]:
        """Return all chunks joined with their parent document's filename.

        The JOIN is necessary because chunks only store document_id (an integer),
        but retrieval results need the human-readable filename for citations.
        Fetching everything at once is intentional — the caller (HybridRetriever)
        loads the entire corpus into memory for fast in-process cosine similarity
        and BM25 scoring, avoiding per-query DB round trips.
        """
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.id,               -- chunk's primary key, used as chunk_id in RetrievalResult
                    c.document_id,      -- which document this chunk belongs to
                    c.chunk_index,      -- position within document
                    c.text,             -- raw prose — shown to LLM as evidence and in citation snippets
                    c.normalized_text,  -- pre-processed text — tokenized for BM25 in _get_corpus()
                    c.section_title,    -- heading — used for section_title_boost in _fuse()
                    c.page_start,       -- used to build citation label e.g. "S1 p.3"
                    c.page_end,         -- used to build citation label e.g. "S1 p.3-4"
                    c.embedding,        -- JSON string — deserialized to np.ndarray in _get_corpus()
                    d.filename          -- "attention.pdf" — chunks table only stores document_id,
                                        -- so we JOIN to get the human-readable name for citations
                FROM chunks c
                JOIN documents d ON d.id = c.document_id  -- attach filename from documents to each chunk row
                ORDER BY c.id          -- stable ordering ensures the corpus cache is consistent across reloads
                """
            ).fetchall()               # loads all rows into memory at once — intentional, HybridRetriever needs them all

    def reset(self) -> None:
        """Delete all documents and chunks — used by the /reset endpoint.

        Chunks must be deleted before documents because of the foreign key constraint.
        _cache_version is incremented so HybridRetriever discards its cached corpus
        and returns an empty result set on the next query.
        """
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks")    # must go first — chunks.document_id references documents.id
                                                        # deleting documents first would violate the FOREIGN KEY constraint
            connection.execute("DELETE FROM documents") # safe now that no chunks reference these rows
        self._cache_version += 1                        # corpus is now empty — next query rebuilds with zero chunks

    @property
    def cache_version(self) -> int:
        return self._cache_version  # @property makes this read-only from outside: storage.cache_version works,
                                    # storage.cache_version = 5 raises AttributeError — prevents accidental mutation
