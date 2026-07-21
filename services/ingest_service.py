"""
PDF ingestion service.

This module provides :class:`IngestService`, which orchestrates the full
ingestion pipeline for a PDF file:

1. **Load** — extract per-page text using :class:`~core.document_loader.DocumentLoader`.
2. **Chunk** — split each page's text into overlapping word windows via
   :class:`~core.chunking.RecursiveChunk`.
3. **Embed** — convert all chunks into dense vectors with
   :meth:`~core.rag_engine.RAGEngine.embed_chunks`.
4. **Store** — persist the document record and all chunk embeddings in the
   Neon vector store via :class:`~core.vector_store.NeonVectorStore`.

The service returns an :class:`~core.models.IngestResult` summary that is
surfaced to API callers through the ``/documents/upload`` endpoint.
"""

from pathlib import Path

from core.chunking import RecursiveChunk
from core.rag_engine import RAGEngine
from core.document_loader import DocumentLoader
from core.models import IngestResult
from core.vector_store import NeonVectorStore
from services.agent_completion import agent_complete


class IngestService:
    """Orchestrates the end-to-end PDF ingestion pipeline.

    Coordinates :class:`~core.document_loader.DocumentLoader`,
    :class:`~core.chunking.RecursiveChunk`,
    :class:`~core.rag_engine.RAGEngine`, and
    :class:`~core.vector_store.NeonVectorStore` to ingest a PDF document
    into the vector store in a single call.
    """

    @staticmethod
    def ingest_pdf(
        file_path: str,
        original_file_name: str | None = None,
    ) -> IngestResult:
        """Ingest a PDF file into the Neon vector store.

        Extracts text page-by-page, chunks each page independently so that
        page-number metadata is preserved per chunk, embeds all chunks in a
        single batch call, and persists everything to the database.

        Args:
            file_path: Path to the PDF file on disk (may be a temporary file
                created during upload).
            original_file_name: The original filename provided by the user.
                When ``None``, the basename of ``file_path`` is used.

        Returns:
            An :class:`~core.models.IngestResult` containing the new
            ``DOCUMENT_ID``, the stored ``file_name``, and counts for
            ``page_count`` and ``chunk_count``.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist (propagated
                from :class:`~core.document_loader.DocumentLoader`).
            ValueError: If the PDF contains no extractable text, or if
                the chunking step produces no chunks.

        Example::

            result = IngestService.ingest_pdf(
                "/tmp/report_abc123.pdf",
                original_file_name="annual_report_2024.pdf",
            )
            print(f"Ingested {result.chunk_count} chunks as document #{result.DOCUMENT_ID}")
        """
        path = Path(file_path)
        file_name = original_file_name if original_file_name else path.name

        pages = DocumentLoader.load_pdf_pages(file_path)

        if not pages:
            raise ValueError("No extractable text found in PDF.")

        all_chunks: list[str] = []
        page_numbers: list[int] = []

        # Chunk each page independently to retain page-number associations.
        for page in pages:
            page_chunks = RecursiveChunk.chunk(page["text"])
            for chunk in page_chunks:
                all_chunks.append(chunk)
                page_numbers.append(page["page_number"])

        if not all_chunks:
            raise ValueError("No chunks created from PDF text.")

        first_page_text = "\n".join([page["text"] for page in pages[:2]])[:1000] if pages else ""

        summary_prompt = [
            {
                "role": "system",
                "content": (
                "Provide a single sentence summarizing what this document is about "
                "based on it's first page. Keept it under 20 words."
                "Ignore version control tables, version numbers, revision history, "
                "release dates, or author credits. Return plain text only"
                )
            },
            {
                "role": "user",
                "content": f"Document text sample:\n{first_page_text}"
            }
        ]

        # Embed all chunks in a single batched API call.
        embeddings = RAGEngine.embed_chunks(all_chunks)

        completion = agent_complete(summary_prompt, is_grading=True)
        doc_summary = completion.choices[0].message.content.strip()

        # Create the parent document record, then insert all chunks.
        DOCUMENT_ID = NeonVectorStore.create_document(file_name=file_name, summary=doc_summary)
        NeonVectorStore.insert_chunks(
            DOCUMENT_ID=DOCUMENT_ID,
            chunks=all_chunks,
            embeddings=embeddings,
            page_numbers=page_numbers,
        )

        return IngestResult(
            DOCUMENT_ID=DOCUMENT_ID,
            file_name=file_name,
            page_count=len(pages),
            chunk_count=len(all_chunks),
        )
