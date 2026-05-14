from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.pdf_utils import ExtractedPage


# A single chunk ready for embedding. page_start/page_end track which PDF pages
# the text came from — used later to build citation labels like "[S1 p.3-4]".
# A chunk can span multiple pages when short pages are merged in chunk_pages().
@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    section_title: str | None = None


# Splits on whitespace that follows a sentence-ending punctuation mark.
# Lookbehind (?<=[.!?]) means the punctuation stays attached to the sentence
# before the split — "Hello. World" → ["Hello.", "World"], not ["Hello", ". World"].
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(pages: list[ExtractedPage], chunk_size: int, overlap: int) -> list[Chunk]:
    """Merge short pages together, then split long text into overlapping chunks.

    Why merge pages first?
    A single PDF page of academic text is often shorter than chunk_size (1100 chars).
    Embedding each page individually would create tiny, low-context chunks that produce
    weak vectors. Merging adjacent short pages into one buffer before splitting ensures
    every chunk is dense with information.

    The buffer accumulates pages until adding the next page would exceed chunk_size.
    At that point the buffer is flushed through split_text() and reset.
    The section_title carried forward is always the FIRST heading seen in the buffer
    (buffer_section or page.section_title) so the chunk's heading reflects where the
    content started, not where it ended.
    """
    chunks: list[Chunk] = []
    buffer = ""               # accumulated text across pages not yet split
    buffer_pages: list[int] = []
    buffer_section: str | None = None

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        # Try adding this page to the existing buffer
        candidate = f"{buffer}\n\n{page_text}".strip() if buffer else page_text

        if len(candidate) <= chunk_size:
            # Page fits — keep accumulating, don't split yet
            buffer = candidate
            buffer_pages.append(page.page_number)
            buffer_section = buffer_section or page.section_title  # keep first heading
            continue

        # Page does not fit — flush whatever is in the buffer first
        if buffer:
            chunks.extend(split_text(buffer, buffer_pages, buffer_section, chunk_size, overlap))
            # Start a fresh buffer with the current page
            buffer = page_text
            buffer_pages = [page.page_number]
            buffer_section = page.section_title
        else:
            # Buffer was empty — this single page is already too long on its own
            chunks.extend(split_text(page_text, [page.page_number], page.section_title, chunk_size, overlap))
            buffer = ""
            buffer_pages = []
            buffer_section = None

    # Flush any remaining text in the buffer after the last page
    if buffer:
        chunks.extend(split_text(buffer, buffer_pages, buffer_section, chunk_size, overlap))

    return chunks


def split_text(
    text: str,
    pages: list[int],
    section_title: str | None,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split a block of text into overlapping chunks at sentence boundaries.

    Why sentence boundaries instead of fixed character positions?
    Embedding models encode meaning — cutting mid-sentence breaks the semantic unit
    and produces a blurry, misleading vector. Splitting at sentence ends keeps each
    chunk semantically coherent.

    Why overlap?
    A concept that straddles the boundary between two chunks (e.g. a definition that
    starts at the end of one chunk and concludes at the start of the next) would be
    invisible to retrieval without overlap. Keeping the last `overlap` characters
    of a chunk as the start of the next ensures such concepts appear fully in at
    least one chunk.

    Edge case — sentence longer than chunk_size:
    Some academic sentences are extremely long (equations, enumerations). If a single
    sentence exceeds chunk_size it cannot be split at a sentence boundary, so it is
    hard-split into chunk_size windows with overlap, the same way a buffer would be.
    """
    # Split the entire text block into individual sentences
    sentences = SENTENCE_BOUNDARY.split(text)
    if not sentences:
        return []

    result: list[Chunk] = []
    current = ""  # sentences accumulated for the chunk being built

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Try appending this sentence to the current accumulator
        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= chunk_size:
            # Sentence fits — keep building the current chunk
            current = candidate
            continue

        # Adding this sentence would exceed chunk_size — emit the current chunk
        if current:
            result.append(
                Chunk(
                    text=current,
                    page_start=min(pages),
                    page_end=max(pages),
                    section_title=section_title,
                )
            )
            # Carry the last `overlap` characters into the next chunk for continuity
            current = current[-overlap:].strip()

        if len(sentence) > chunk_size:
            # Single sentence is too long even on its own — hard-split with overlap
            start = 0
            while start < len(sentence):
                end = min(start + chunk_size, len(sentence))
                piece = sentence[start:end].strip()
                if piece:
                    result.append(
                        Chunk(
                            text=piece,
                            page_start=min(pages),
                            page_end=max(pages),
                            section_title=section_title,
                        )
                    )
                # Move forward by chunk_size but step back by overlap for continuity.
                # max(..., start + 1) prevents an infinite loop if overlap >= chunk_size.
                start = max(end - overlap, start + 1)
            current = ""  # hard-split consumed the sentence; start fresh
        else:
            # Sentence fits on its own — start a new accumulator with it
            current = f"{current} {sentence}".strip() if current else sentence

    # Emit any remaining text that didn't fill a full chunk
    if current:
        result.append(
            Chunk(
                text=current,
                page_start=min(pages),
                page_end=max(pages),
                section_title=section_title,
            )
        )
    return result
