from __future__ import annotations

from datetime import UTC, datetime  # UTC timestamp for collision-safe upload filenames
from pathlib import Path            # cross-platform file path handling
import re                           # sentence splitting in _citation_snippet

from fastapi import UploadFile      # FastAPI's type for multipart file uploads

from app.config import settings     # chunk_size, overlap, thresholds, model names
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
    """Orchestrates the full ingestion and query pipelines.

    Holds references to all major components and wires them together.
    All heavy logic lives in the individual service modules — RagService
    is the coordinator, not the implementer.
    """

    def __init__(self, storage: Storage, mistral_client: MistralClient) -> None:
        self.storage = storage                   # SQLite store for documents and chunks
        self.mistral_client = mistral_client     # async HTTP wrapper for Mistral API
        self.retriever = HybridRetriever(storage) # BM25 + cosine + HyDE hybrid search

    async def ingest_files(self, files: list[UploadFile]) -> list[IngestionFileResult]:
        """Process each uploaded PDF and store its chunks + embeddings in SQLite.

        Processes files one at a time (not in parallel) to avoid overwhelming
        the Mistral embedding API with too many concurrent requests.
        Each file is fully ingested before moving to the next.
        """
        results: list[IngestionFileResult] = []
        for upload in files:
            # validate early — reject non-PDFs before touching disk
            if not upload.filename or not upload.filename.lower().endswith(".pdf"):
                raise ValueError(f"Only PDF files are supported: {upload.filename!r}")

            # build a timestamped path so uploading the same file twice doesn't overwrite
            # e.g. data/uploads/20260417055322507750_attention.pdf
            destination = self._build_upload_path(upload.filename)
            contents = await upload.read()           # read the full file bytes from the HTTP request
            destination.write_bytes(contents)        # persist to disk before processing

            # extract one ExtractedPage per PDF page
            pages = extract_pdf_pages(destination)
            # drop bibliography/reference pages — they contaminate search results
            # with author names and citation markers rather than actual content
            pages = [page for page in pages if not looks_like_references_page(page.text)]

            # split pages into overlapping text chunks at sentence boundaries
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                # scanned/image PDFs produce no text — fail loudly rather than silently storing nothing
                raise ValueError(f"No readable text was extracted from {upload.filename}.")

            # embed all chunk texts in one call (batched internally by embed_texts)
            # returns list[list[float]] — one vector per chunk, same order
            embeddings = await self.mistral_client.embed_texts([chunk.text for chunk in chunks])

            # insert the document record first to get its auto-generated ID
            # all chunks reference this ID via document_id foreign key
            document_id = self.storage.insert_document(upload.filename, str(destination))

            # insert all chunks as a generator — storage materializes it in one transaction
            self.storage.insert_chunks(
                {
                    "document_id": document_id,
                    "chunk_index": index,              # 0, 1, 2 ... position within document
                    "text": chunk.text,                # raw prose shown to the LLM
                    "normalized_text": normalize(chunk.text),  # lowercased tokens for BM25
                    "section_title": chunk.section_title,      # heading for retrieval boost
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "embedding": embeddings[index],    # list[float] matched by position
                }
                for index, chunk in enumerate(chunks)  # enumerate gives (0,chunk0),(1,chunk1)...
            )
            results.append(
                IngestionFileResult(
                    filename=upload.filename,
                    document_id=document_id,
                    chunks_created=len(chunks),
                )
            )
        return results  # one IngestionFileResult per uploaded file

    async def answer_query(self, query: str, top_k: int | None = None) -> QueryResponse:
        """Run the full query pipeline and return a cited answer.

        Pipeline order:
          1. detect_intent      — short-circuit chitchat and restricted queries
          2. rewrite_query      — add keyword suffix for BM25
          3. embed query        — vector for semantic search
          4. HyDE embedding     — hypothetical answer vector for better recall
          5. retrieve           — hybrid BM25 + cosine + HyDE search
          6. evidence gate      — refuse if top score below threshold
          7. build citations    — format source labels and snippets
          8. LLM generation     — generate answer grounded in evidence
          9. hallucination check — reject if answer cites unsupported claims
        """
        intent = detect_intent(query)

        # restricted: PII / medical / legal — refuse immediately, no retrieval
        if intent == "restricted":
            return QueryResponse(
                intent="restricted",
                answer=(
                    "I can only answer grounded questions about the uploaded PDFs. "
                    "I won't help with PII extraction or provide legal or medical advice."
                ),
                insufficient_evidence=True,
            )

        # chitchat: greetings, "who are you" etc — fixed reply, no retrieval
        if intent == "chitchat":
            return QueryResponse(
                intent="chitchat",
                answer="Hello. Ask a question about the uploaded PDFs and I'll answer with citations.",
            )

        # knowledge query — proceed through the full retrieval pipeline
        rewritten_query = rewrite_query(query)  # "query | keywords: term1, term2..."

        # embed the rewritten query — [0] because embed_texts returns list[list[float]]
        # and we only passed one text
        query_embedding = (await self.mistral_client.embed_texts([rewritten_query]))[0]

        # HyDE: ask LLM to write a hypothetical answer, then embed it
        # returns None if hyde_enabled=False or if the LLM call fails
        hyde_embedding = await self._hyde_embedding(query)

        # run hybrid retrieval — wide funnel (12+12+12) → fuse → dedup → MMR → top-k
        retrieved = self.retriever.retrieve(
            rewritten_query,
            query_embedding,
            hyde_embedding=hyde_embedding,
            top_k=top_k,
        )

        # evidence gate: if no chunks retrieved or best score below threshold → insufficient evidence
        # avoids calling the LLM when the knowledge base has no relevant content
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

        # build Citation objects from each retrieved chunk
        # _source_label → "S1 p.3" or "S1 p.3-4" for multi-page chunks
        # _citation_snippet → the 2 sentences from the chunk most relevant to the query
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

        # call the LLM with the evidence-grounded prompt
        # temperature=0.1 → near-deterministic, consistent answers
        answer = await self.mistral_client.chat(
            messages=self._build_messages(query, retrieved),
            temperature=0.1,
        )

        # post-hoc hallucination check: if >1/3 of answer sentences have zero
        # evidence term coverage → reject the answer entirely
        if hallucination_check(answer, retrieved):
            answer = "insufficient evidence"

        return QueryResponse(
            intent="knowledge",
            rewritten_query=rewritten_query,
            answer=answer,
            citations=[] if answer == "insufficient evidence" else citations,  # no citations if rejected
            insufficient_evidence=answer == "insufficient evidence",
            debug={
                "top_score": round(retrieved[0].fused_score, 4),
                "semantic_score": round(retrieved[0].semantic_score, 4),
                "keyword_score": round(retrieved[0].keyword_score, 4),
            },
        )

    def _build_messages(self, query: str, retrieved) -> list[dict]:
        """Format the system prompt + user prompt for the LLM.

        System prompt instructs the LLM to:
          - use only the provided evidence (no hallucination)
          - include exact source labels like [S1 p.5] for every claim
          - say "insufficient evidence" if evidence is weak or contradictory
          - adapt format: bullets/table for list queries, prose for factual queries

        User prompt combines the original query with all retrieved evidence chunks,
        each labelled with its source file and page numbers.

        Evidence format example:
          [S1 p.3] file=attention.pdf pages=3-3
          The attention mechanism computes a weighted sum of values...

          [S2 p.5] file=attention.pdf pages=5-6
          Multi-head attention allows the model to attend...
        """
        intent_template = (
            "When the user asks for a list or table, answer in Markdown bullets or a compact Markdown table. "
            "For factual questions, answer in concise prose."
        )
        # format each chunk as a labelled evidence block
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
            {"role": "system", "content": system_prompt},  # instructions to the LLM
            {"role": "user", "content": user_prompt},      # the actual question + evidence
        ]

    async def _hyde_embedding(self, query: str) -> list[float] | None:
        """Generate a HyDE embedding — embed a hypothetical answer instead of the query.

        Why? Query embeddings and document embeddings often live in different
        regions of the vector space (question style vs answer style). A hypothetical
        answer written in document style retrieves better matches via cosine similarity.

        temperature=0.0 → fully deterministic generation (factual style, no creativity)
        Returns None if HyDE is disabled in config or if the LLM call fails —
        retrieval continues without it rather than crashing.
        """
        if not settings.hyde_enabled:
            return None   # feature disabled — skip the extra LLM call

        try:
            # ask the LLM to write what a relevant document passage would look like
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
                temperature=0.0,  # deterministic — we want a consistent factual passage
            )
            # embed the hypothetical passage — [0] because we passed a single text
            return (await self.mistral_client.embed_texts([hypothetical]))[0]
        except MistralAPIError:
            # HyDE failure is non-fatal — retrieval continues with just the query embedding
            return None

    def _source_label(self, index: int, item) -> str:
        """Build a citation label like "S1 p.3" or "S2 p.5-6".

        S{index+1} — source number (1-based so first source is S1, not S0)
        p.{start}  — single page chunk
        p.{start}-{end} — multi-page chunk
        """
        if item.page_start == item.page_end:
            return f"S{index + 1} p.{item.page_start}"       # e.g. "S1 p.3"
        return f"S{index + 1} p.{item.page_start}-{item.page_end}"  # e.g. "S2 p.5-6"

    def _build_upload_path(self, filename: str) -> Path:
        """Build a timestamped, space-free path for the uploaded file.

        Timestamp format: YYYYMMDDHHMMSSffffff (microseconds)
        Example: 20260417055322507750_attention.pdf

        Why timestamp prefix?
        Uploading "attention.pdf" twice would overwrite the first file.
        The timestamp makes every upload path unique — both files are kept.

        Why replace spaces?
        Filenames with spaces cause issues in some shell tools and URLs.
        "my paper.pdf" → "my_paper.pdf"
        """
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")   # UTC to avoid timezone issues
        safe_name = Path(filename).name.replace(" ", "_")        # strip directory parts + sanitize spaces
        return settings.uploads_dir / f"{stamp}_{safe_name}"

    def _citation_snippet(self, query: str, text: str) -> str:
        """Extract the 2 most query-relevant sentences from a chunk for the citation preview.

        Why 2 sentences?
        The full chunk text (up to 1100 chars) is too long for a citation snippet
        shown in the UI. Two sentences give enough context without overwhelming the reader.

        Ranking: sentences are scored by how many query terms they contain.
        Tie-break: longer sentence preferred (more context).

        Example:
          query:    "how does attention work"
          chunk:    "Transformers use attention. The attention mechanism computes
                     weighted sums. This was proposed in 2017."
          query terms: {"attention", "work"}
          scores:
            "Transformers use attention."              → 1 match ("attention")
            "The attention mechanism computes..."      → 1 match ("attention")
            "This was proposed in 2017."               → 0 matches
          top 2 (by match count, then length):
            → "The attention mechanism computes..." + "Transformers use attention."
        """
        # split chunk into sentences at .  !  ? boundaries
        sentences = [
            " ".join(sentence.split())   # collapse internal whitespace
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()          # skip empty strings from split
        ]
        if not sentences:
            return " ".join(text.split())   # fallback: return cleaned full text

        query_terms = set(normalize(query).split())  # tokenized query terms for matching
        scored = sorted(
            sentences,
            key=lambda sentence: (
                len(query_terms & set(normalize(sentence).split())),  # primary: term overlap count
                len(sentence),                                         # tiebreak: longer = more context
            ),
            reverse=True,   # highest overlap first
        )
        top_sentences = scored[:2]               # take the 2 best sentences
        return " ".join(top_sentences).strip()
