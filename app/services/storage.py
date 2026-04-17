from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                """
            )

    def insert_document(self, filename: str, stored_path: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO documents (filename, stored_path) VALUES (?, ?)",
                (filename, stored_path),
            )
            return int(cursor.lastrowid)

    def insert_chunks(self, rows: Iterable[dict]) -> None:
        payload = [
            (
                row["document_id"],
                row["chunk_index"],
                row["text"],
                row["normalized_text"],
                row["page_start"],
                row["page_end"],
                json.dumps(row["embedding"]),
            )
            for row in rows
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks (
                    document_id, chunk_index, text, normalized_text, page_start, page_end, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def fetch_chunks(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.normalized_text,
                    c.page_start,
                    c.page_end,
                    c.embedding,
                    d.filename
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.id
                """
            ).fetchall()

    def reset(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
