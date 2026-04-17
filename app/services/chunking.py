from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.pdf_utils import ExtractedPage


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    section_title: str | None = None


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(pages: list[ExtractedPage], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer = ""
    buffer_pages: list[int] = []
    buffer_section: str | None = None

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue
        candidate = f"{buffer}\n\n{page_text}".strip() if buffer else page_text
        if len(candidate) <= chunk_size:
            buffer = candidate
            buffer_pages.append(page.page_number)
            buffer_section = buffer_section or page.section_title
            continue

        if buffer:
            chunks.extend(split_text(buffer, buffer_pages, buffer_section, chunk_size, overlap))
            buffer = page_text
            buffer_pages = [page.page_number]
            buffer_section = page.section_title
        else:
            chunks.extend(split_text(page_text, [page.page_number], page.section_title, chunk_size, overlap))
            buffer = ""
            buffer_pages = []
            buffer_section = None

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
    sentences = SENTENCE_BOUNDARY.split(text)
    if not sentences:
        return []

    result: list[Chunk] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            result.append(
                Chunk(
                    text=current,
                    page_start=min(pages),
                    page_end=max(pages),
                    section_title=section_title,
                )
            )
            current = current[-overlap:].strip()
        if len(sentence) > chunk_size:
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
                start = max(end - overlap, start + 1)
            current = ""
        else:
            current = f"{current} {sentence}".strip() if current else sentence

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
