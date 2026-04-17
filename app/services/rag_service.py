from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re

from fastapi import UploadFile

from app.config import settings
from app.schemas import Citation, IngestionFileResult, QueryResponse
from app.services.chunking import chunk_pages
from app.services.mistral_client import MistralAPIError, MistralClient
from app.services.pdf_utils import extract_pdf_pages, looks_like_references_page
from app.services.retrieval import (
    HybridRetriever,
    detect_intent,
    hallucination_check,
    normalize,
    rewrite_query,
)
from app.services.storage import Storage


class RagService:
    def __init__(self, storage: Storage, mistral_client: MistralClient) -> None:
        self.storage = storage
        self.mistral_client = mistral_client
        self.retriever = HybridRetriever(storage)

    async def ingest_files(self, files: list[UploadFile]) -> list[IngestionFileResult]:
        results: list[IngestionFileResult] = []
        for upload in files:
            if not upload.filename or not upload.filename.lower().endswith(".pdf"):
                raise ValueError(f"Only PDF files are supported: {upload.filename!r}")

            destination = self._build_upload_path(upload.filename)
            contents = await upload.read()
            destination.write_bytes(contents)

            pages = extract_pdf_pages(destination)
            pages = [page for page in pages if not looks_like_references_page(page.text)]
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                raise ValueError(f"No readable text was extracted from {upload.filename}.")
            embeddings = await self.mistral_client.embed_texts([chunk.text for chunk in chunks])
            document_id = self.storage.insert_document(upload.filename, str(destination))
            self.storage.insert_chunks(
                {
                    "document_id": document_id,
                    "chunk_index": index,
                    "text": chunk.text,
                    "normalized_text": normalize(chunk.text),
                    "section_title": chunk.section_title,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "embedding": embeddings[index],
                }
                for index, chunk in enumerate(chunks)
            )
            results.append(
                IngestionFileResult(
                    filename=upload.filename,
                    document_id=document_id,
                    chunks_created=len(chunks),
                )
            )
        return results

    async def answer_query(self, query: str, top_k: int | None = None) -> QueryResponse:
        intent = detect_intent(query)
        if intent == "restricted":
            return QueryResponse(
                intent="restricted",
                answer=(
                    "I can only answer grounded questions about the uploaded PDFs. "
                    "I won't help with PII extraction or provide legal or medical advice."
                ),
                insufficient_evidence=True,
            )

        if intent == "chitchat":
            return QueryResponse(
                intent="chitchat",
                answer="Hello. Ask a question about the uploaded PDFs and I’ll answer with citations.",
            )

        rewritten_query = rewrite_query(query)
        query_embedding = (await self.mistral_client.embed_texts([rewritten_query]))[0]
        hyde_embedding = await self._hyde_embedding(query)
        retrieved = self.retriever.retrieve(
            rewritten_query,
            query_embedding,
            hyde_embedding=hyde_embedding,
            top_k=top_k,
        )
        if not retrieved or retrieved[0].fused_score < settings.min_evidence_score:
            return QueryResponse(
                intent="knowledge",
                rewritten_query=rewritten_query,
                answer="insufficient evidence",
                insufficient_evidence=True,
                debug={
                    "top_score": retrieved[0].fused_score if retrieved else 0.0,
                    "threshold": settings.min_evidence_score,
                },
            )

        citations = [
            Citation(
                label=self._source_label(index, item),
                document_id=item.document_id,
                filename=item.filename,
                chunk_id=item.chunk_id,
                section_title=item.section_title,
                page_start=item.page_start,
                page_end=item.page_end,
                score=round(item.fused_score, 4),
                snippet=self._citation_snippet(query, item.text),
            )
            for index, item in enumerate(retrieved)
        ]

        answer = await self.mistral_client.chat(
            messages=self._build_messages(query, retrieved),
            temperature=0.1,
        )
        if hallucination_check(answer, retrieved):
            answer = "insufficient evidence"

        return QueryResponse(
            intent="knowledge",
            rewritten_query=rewritten_query,
            answer=answer,
            citations=[] if answer == "insufficient evidence" else citations,
            insufficient_evidence=answer == "insufficient evidence",
            debug={
                "top_score": round(retrieved[0].fused_score, 4),
                "semantic_score": round(retrieved[0].semantic_score, 4),
                "keyword_score": round(retrieved[0].keyword_score, 4),
            },
        )

    def _build_messages(self, query: str, retrieved) -> list[dict]:
        intent_template = (
            "When the user asks for a list or table, answer in Markdown bullets or a compact Markdown table. "
            "For factual questions, answer in concise prose."
        )
        evidence = "\n\n".join(
            [
                (
                    f"[{self._source_label(index, item)}] file={item.filename} pages={item.page_start}-{item.page_end}\n"
                    f"{item.text}"
                )
                for index, item in enumerate(retrieved)
            ]
        )
        system_prompt = (
            "You are a RAG assistant. Use only the provided evidence. "
            "If the evidence is weak, incomplete, or contradictory, answer exactly: insufficient evidence. "
            "For each material claim, include one or more exact source labels copied from the evidence, "
            "such as [S1 p.5] or [S2 p.9-10]. Do not invent citation labels. "
            f"{intent_template}"
        )
        user_prompt = f"Question: {query}\n\nEvidence:\n{evidence}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _hyde_embedding(self, query: str) -> list[float] | None:
        if not settings.hyde_enabled:
            return None

        try:
            hypothetical = await self.mistral_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a short hypothetical answer passage that would likely appear in a relevant document. "
                            "Be factual in style, 3-5 sentences, and focus on the likely answer terms."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            return (await self.mistral_client.embed_texts([hypothetical]))[0]
        except MistralAPIError:
            return None

    def _source_label(self, index: int, item) -> str:
        if item.page_start == item.page_end:
            return f"S{index + 1} p.{item.page_start}"
        return f"S{index + 1} p.{item.page_start}-{item.page_end}"

    def _build_upload_path(self, filename: str) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        safe_name = Path(filename).name.replace(" ", "_")
        return settings.uploads_dir / f"{stamp}_{safe_name}"

    def _citation_snippet(self, query: str, text: str) -> str:
        sentences = [
            " ".join(sentence.split())
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        if not sentences:
            return " ".join(text.split())

        query_terms = set(normalize(query).split())
        scored = sorted(
            sentences,
            key=lambda sentence: (
                len(query_terms & set(normalize(sentence).split())),
                len(sentence),
            ),
            reverse=True,
        )
        top_sentences = scored[:2]
        return " ".join(top_sentences).strip()
