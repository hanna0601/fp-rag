from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


# One page extracted from a PDF. section_title is detected from the page's
# first lines and later used in retrieval to boost chunks under a relevant heading.
@dataclass
class ExtractedPage:
    page_number: int
    text: str
    section_title: str | None = None


def extract_pdf_pages(path: Path) -> list[ExtractedPage]:
    """Read every page of a PDF and return cleaned ExtractedPage objects.

    pypdf is a pure-Python PDF parser — no external binaries needed.
    Pages are 1-indexed (start=1) so page numbers match what readers see in the document.
    Empty pages (e.g. blank separator pages) are silently skipped — they would
    produce empty chunks that waste embedding API calls and pollute search results.
    """
    reader = PdfReader(str(path))
    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        # extract_text() returns None for image-only pages (scanned PDFs without OCR)
        text = page.extract_text() or ""
        cleaned = normalize_whitespace(text)
        if cleaned:  # skip truly empty pages
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text=cleaned,
                    section_title=extract_section_title(cleaned),
                )
            )
    return pages


def normalize_whitespace(text: str) -> str:
    """Collapse noisy whitespace that pypdf commonly produces from PDF layout encoding.

    PDF files store text as positioned glyphs, not flowing prose. When pypdf
    reconstructs the text stream it often introduces:
      - \x00 null bytes from encoding artifacts
      - mixed \r\n / \r line endings from cross-platform PDFs
      - runs of spaces/tabs where the original had visual spacing
      - excessive blank lines between paragraphs or around figures

    Cleaning these now means the chunker and embedding model see clean prose,
    not layout noise that would degrade semantic similarity scores.
    """
    text = text.replace("\x00", " ")         # null bytes → space (encoding artifact)
    text = re.sub(r"\r\n?", "\n", text)      # normalize line endings to \n
    text = re.sub(r"[ \t]+", " ", text)      # collapse runs of spaces/tabs to one space
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse 3+ blank lines to one paragraph break
    return text.strip()


def looks_like_references_page(text: str) -> bool:
    """Detect bibliography/reference pages so they can be excluded from the knowledge base.

    Why exclude them? A references page is full of author names, years, journal
    titles, and technical terms borrowed from many papers. Without exclusion, these
    pages would match almost any academic query — returning chunks like
    "[47] Vaswani et al. 2017. Attention is all you need." instead of actual content.

    Detection uses a heuristic combination of signals rather than a single keyword,
    because a paper body can mention "references" or contain "[1]" without being
    a reference page. We only filter when multiple signals agree:

      signal_hits   — how many of the five header keywords appear in the page text
      bracketed_refs — count of citation markers like [1], [23], [104]
      year_hits      — count of 4-digit years (1900-2099), common in citations

    Decision rule:
      - 2+ header keywords → almost certainly a reference section
      - 1 keyword + 6+ bracketed refs + 6+ years → bibliography-style page body
    """
    lowered = text.lower()
    reference_signals = [
        "references",       # standard section header in academic papers
        "bibliography",     # alternative header in some fields
        "acknowledgements", # often appears on the same page as references
        "arxiv preprint",   # common footer on preprint reference lists
        "proceedings of",   # conference citation format marker
    ]
    signal_hits = sum(1 for signal in reference_signals if signal in lowered)
    bracketed_refs = len(re.findall(r"\[\d+\]", text))               # e.g. [1], [42]
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", text))           # e.g. 1998, 2024
    return signal_hits >= 2 or (signal_hits >= 1 and bracketed_refs >= 6 and year_hits >= 6)


def extract_section_title(text: str) -> str | None:
    """Try to detect the section heading from the top of a page's text.

    Why? Section titles are stored on each chunk and used in retrieval to give
    a small score boost when the query terms match the heading. For example,
    a query about "attention mechanism" scores higher on a chunk whose section
    title is "3. Attention Mechanism" than on an equally dense chunk with no heading.

    Strategy: inspect only the first 6 lines (headings appear at the top of a page)
    and match two common academic heading patterns via regex:

      Pattern 1 — optional leading digits/dots then a capital letter:
        "Introduction", "3. Experiments", "A. Appendix"

      Pattern 2 — numbered subsection then a capital letter:
        "3.1 Self-Attention", "4.2.1 Results"

    Lines longer than 90 characters are skipped — those are prose sentences,
    not headings (headings are typically short labels).

    Returns None if no heading-shaped line is found in the first 6 lines.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:6]:  # headings appear near the top of the page
        if len(line) > 90:  # long lines are prose, not headings
            continue
        # Pattern 1: "Introduction", "3. Method", "A Results"
        if re.fullmatch(r"[\d. ]*[A-Z][A-Za-z0-9 ,:/()\-]{2,}", line):
            return line
        # Pattern 2: "3.1 Self-Attention", "4.2.1 Ablation Study"
        if re.fullmatch(r"\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,:/()\-]{2,}", line):
            return line
    return None
