"""
PDF document loading utilities.

This module provides :class:`DocumentLoader`, a thin wrapper around
:mod:`pypdf` that extracts text content from PDF files.  Two extraction
modes are available:

- **Flat** (:meth:`~DocumentLoader.load_pdf`) — all pages concatenated into
  a single string, suitable for in-memory chunking.
- **Paged** (:meth:`~DocumentLoader.load_pdf_pages`) — per-page dicts
  retaining page-number metadata, required for page-aware chunk storage in
  the vector store.

Pages that yield no extractable text (e.g. scanned images without OCR) are
silently skipped in both modes.
"""

from pathlib import Path

from pypdf import PdfReader


class DocumentLoader:
    """Utility class for extracting text from PDF files using pypdf.

    All methods are static; no instantiation is required.
    """

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract all text from a PDF as a single concatenated string.

        Reads every page of the PDF and joins non-empty page texts with
        newline separators.  Pages that contain no extractable text are
        skipped without raising an error.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A single string containing the text of all extractable pages,
            joined by ``"\\n"``.

        Raises:
            FileNotFoundError: If no file exists at ``file_path``.

        Example::

            text = DocumentLoader.load_pdf("/tmp/report.pdf")
            chunks = RecursiveChunk.chunk(text)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(path)
        page_text: list[str] = []

        for _, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                page_text.append(text)

        return "\n".join(page_text)

    @staticmethod
    def load_pdf_pages(file_path: str) -> list[dict]:
        """Extract text from a PDF retaining per-page metadata.

        Returns one dictionary per extractable page, preserving the
        original page number.  This structured output is required by the
        ingestion pipeline so that chunk-to-page associations can be stored
        in the vector store.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A list of dicts, one per non-empty page, each with the keys:

            - ``"page_number"`` (int): 1-based page index.
            - ``"text"`` (str): Extracted text for that page.

            Pages with no extractable text are omitted.

        Raises:
            FileNotFoundError: If no file exists at ``file_path``.

        Example::

            pages = DocumentLoader.load_pdf_pages("/tmp/report.pdf")
            for page in pages:
                print(page["page_number"], page["text"][:80])
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(path)
        pages: list[dict] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

        return pages