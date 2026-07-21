"""
Domain models for the PDF RAG pipeline.

This module defines the core data transfer objects (DTOs) shared across the
ingestion, retrieval, and chat layers of the application.  All models are
implemented as Python :func:`dataclasses` for lightweight, type-safe
instantiation without external dependencies.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Represents a single chunk retrieved from the vector store.

    Produced by both in-memory cosine similarity search
    (:meth:`~core.rag_engine.RAGEngine.retrieve`) and database-backed searches
    (:meth:`~core.vector_store.NeonVectorStore.similarity_search`,
    :meth:`~core.vector_store.NeonVectorStore.keyword_search`,
    :meth:`~core.vector_store.NeonVectorStore.hybrid_search`).

    Attributes:
        score (float): Relevance score for this chunk relative to the query.
            Interpretation depends on :attr:`retrieval_method`:

            - ``"vector"``  — cosine similarity converted from distance
              (``1 - pgvector_distance``), range roughly ``[0, 1]``.
            - ``"keyword"`` — PostgreSQL ``ts_rank`` score, unbounded positive.
            - ``"hybrid"``  — Reciprocal Rank Fusion (RRF) score, small
              positive float (typically ``< 0.1``).

        chunk_id (int): Zero-based index of the chunk within its document, as
            stored in the ``chunk_index`` database column.
        chunk_text (str): The raw text content of the retrieved chunk.
        page_number (int | None): The source page number within the original
            PDF, or ``None`` if page tracking was not used during ingestion.
        retrieval_method (str): The search strategy that produced this result.
            One of ``"vector"``, ``"keyword"``, ``"hybrid"``, or
            ``"unknown"`` (default).
    """

    score: float
    chunk_id: int
    chunk_text: str
    page_number: int | None = None
    retrieval_method: str = "unknown"

    def label(self) -> str:
        """Return a human-readable citation label for this chunk.

        Includes the page number when available so the LLM can cite the
        exact source location in its answer.

        Returns:
            A string such as ``"Page 3, Chunk 7"`` when page information is
            present, or ``"Chunk 7"`` when it is not.
        """
        if self.page_number:
            return f"Page {self.page_number}, Chunk {self.chunk_id + 1}"

        return f"Chunk {self.chunk_id + 1}"


@dataclass
class IngestResult:
    """Summary returned after successfully ingesting a PDF document.

    Returned by :meth:`~services.ingest_service.IngestService.ingest_pdf` and
    surfaced to API callers via the ``/documents/upload`` endpoint.

    Attributes:
        DOCUMENT_ID (int): The auto-generated primary key assigned to the new
            document row in the ``documents`` table.
        file_name (str): The original filename of the uploaded PDF.
        page_count (int): Number of pages successfully extracted from the PDF.
        chunk_count (int): Total number of text chunks stored in the vector
            store for this document.
    """

    DOCUMENT_ID: int
    file_name: str
    page_count: int
    chunk_count: int


@dataclass
class ChatResult:
    """Container for the final answer, supporting source chunks, and execution metrics.

    Returned by :meth:`~services.graph_services.GraphService.ask_pdf_with_graph`
    and serialised into :class:`~schemas.QueryResponse` by the chat route.

    Attributes:
        answer (str): The LLM-generated answer to the user's question.
        sources (list[RetrievalResult]): The subset of retrieved chunks that
            were selected as context for generating the answer.
        used_rewrite (bool): Whether the search query was dynamically expanded/rewritten.
        is_grounded (bool): Whether the final answer was verified to be factual by the verifier node.
        retrieval_count (int): Total number of chunks retrieved from the store after threshold filtering.
        selected_count (int): Total number of top chunks selected for inclusion in the LLM context.
    """

    answer: str
    sources: list[RetrievalResult]
    used_rewrite: bool = False
    is_grounded: bool = False
    retrieval_count: int = 0
    selected_count: int = 0

