"""PDF and page processing helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader, PdfWriter

from .logging_config import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PagePayload:
    index: int
    data: bytes
    mime_type: str


def split_pdf(pdf_bytes: bytes) -> List[PagePayload]:
    """Split a PDF into per-page payloads."""

    buffer = io.BytesIO(pdf_bytes)
    reader = PdfReader(buffer)
    pages: List[PagePayload] = []
    for idx, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        out_buffer = io.BytesIO()
        writer.write(out_buffer)
        pages.append(PagePayload(index=idx, data=out_buffer.getvalue(), mime_type="application/pdf"))
    LOGGER.info("PDF split into pages", extra={"totalPages": len(pages)})
    return pages


__all__ = ["PagePayload", "split_pdf"]
