from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    section_title: str | None = None


def extract_pdf_pages(path: Path) -> list[ExtractedPage]:
    reader = PdfReader(str(path))
    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = normalize_whitespace(text)
        if cleaned:
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text=cleaned,
                    section_title=extract_section_title(cleaned),
                )
            )
    return pages


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_references_page(text: str) -> bool:
    lowered = text.lower()
    reference_signals = [
        "references",
        "bibliography",
        "acknowledgements",
        "arxiv preprint",
        "proceedings of",
    ]
    signal_hits = sum(1 for signal in reference_signals if signal in lowered)
    bracketed_refs = len(re.findall(r"\[\d+\]", text))
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", text))
    return signal_hits >= 2 or (signal_hits >= 1 and bracketed_refs >= 6 and year_hits >= 6)


def extract_section_title(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:6]:
        if len(line) > 90:
            continue
        if re.fullmatch(r"[\d. ]*[A-Z][A-Za-z0-9 ,:/()\-]{2,}", line):
            return line
        if re.fullmatch(r"\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,:/()\-]{2,}", line):
            return line
    return None
