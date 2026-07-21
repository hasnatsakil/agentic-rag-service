"""
Pydantic request and response schemas for the PDF RAG Chat API.

All schemas inherit from :class:`pydantic.BaseModel` and are used by FastAPI
for automatic request validation, response serialisation, and OpenAPI
documentation generation.

Classes overview:
    - :class:`QueryRequest`          — body for ``POST /chat/query``
    - :class:`QueryResponse`         — response for ``POST /chat/query``
    - :class:`SourceResponse`        — individual source chunk in a query response
    - :class:`UploadResponse`        — response for ``POST /documents/upload``
    - :class:`DocumentResponse`      — single document record
    - :class:`DocumentsListResponse` — list of documents for ``GET /documents``
    - :class:`DeleteDocumentResponse``— response for ``DELETE /documents/{id}``
    - :class:`HealthResponse`        — response for ``GET /health``
"""

from pydantic import BaseModel, Field

from config import settings


class QueryRequest(BaseModel):
    """Request body for the ``POST /chat/query`` endpoint.

    All fields except ``session_id`` and ``question`` have defaults sourced from application
    settings, so callers only need to provide the question and session ID.

    Attributes:
        session_id (str): Unique session identifier for chat history tracking. Required.
        question (str): The user's natural-language question. Required.
        SEARCH_K (int): Number of candidate chunks to retrieve from the
            vector store before grading. Defaults to ``settings.SEARCH_K``.
        GRADE_K (int): Number of top retrieved chunks to grade for relevance. Defaults to ``settings.GRADE_K``.
        ANSWER_K (int): Number of top-ranked chunks forwarded to the LLM
            for answer generation. Defaults to ``settings.ANSWER_K``.
        MIN_SCORE (float): Minimum retrieval score; chunks below this are
            filtered out. Defaults to ``settings.MIN_SCORE``.
        MAX_CONTEXT_CHARS (int): Maximum total characters of context sent
            to the LLM. Prevents prompt overflow. Defaults to
            ``settings.MAX_CONTEXT_CHARS``.
        use_llm_rerank (bool): Whether to use a Pass 2 LLM-as-a-Judge node to rerank chunks before answering.
    """

    session_id: str = Field(
        ...,
        description="Unique session identifier for chat history tracking",
    )
    question: str = Field(
        ...,
        description="Question to ask about the document",
    )
    SEARCH_K: int = Field(
        settings.SEARCH_K,
        description="Number of chunks to retrieve from vector store for context",
    )
    GRADE_K: int = Field(
        settings.GRADE_K,
        description="Number of top retrieved chunks to grade for relevance",
    )
    ANSWER_K: int = Field(
        settings.ANSWER_K,
        description="Number of chunks sent to LLM",
    )
    MIN_SCORE: float = Field(
        settings.MIN_SCORE,
        description="Minimum cosine similarity score",
    )
    MAX_CONTEXT_CHARS: int = Field(
        settings.MAX_CONTEXT_CHARS,
        description="Maximum total characters of context to send to LLM",
    )
    use_llm_rerank: bool = Field(
        False,
        description="Whether to use an LLM to rerank retrieved chunks before answering"
    )


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint.

    Attributes:
        status (str): Overall service status, e.g. ``"ok"`` or ``"degraded"``.
        message (str): Human-readable description of the health state.
    """

    status: str
    message: str


class UploadResponse(BaseModel):
    """Response body for the ``POST /documents/upload`` endpoint.

    Attributes:
        message (str): Human-readable confirmation message.
        DOCUMENT_ID (int): The auto-generated ID assigned to the ingested document.
        file_name (str): The stored filename of the uploaded PDF.
        page_count (int): Number of pages successfully extracted.
        chunk_count (int): Number of text chunks stored in the vector store.
    """

    message: str
    DOCUMENT_ID: int
    file_name: str
    page_count: int
    chunk_count: int


class SourceResponse(BaseModel):
    """Serialised representation of a single retrieved source chunk.

    Used as an element of :attr:`QueryResponse.sources`.

    Attributes:
        label (str): Human-readable citation, e.g. ``"Page 3, Chunk 7"``.
        score (float): Retrieval relevance score for this chunk.
        chunk_id (int): Zero-based index of the chunk within its document.
        page_number (int | None): Source page in the original PDF, or
            ``None`` if page tracking was not used.
        chunk_text (str): The raw text content of the chunk.
    """

    label: str
    score: float
    chunk_id: int
    page_number: int | None
    chunk_text: str


class QueryDebugResponse(BaseModel):
    """Execution metrics and debug flags returned in the query response.

    Attributes:
        used_rewrite (bool): True if the query was expanded/rewritten during state graph execution.
        is_grounded (bool): True if the answer passed the factual verifier grade.
        retrieval_count (int): Number of chunks retrieved matching MIN_SCORE.
        selected_count (int): Number of chunks selected for the LLM prompt.
    """

    used_rewrite: bool
    is_grounded: bool
    retrieval_count: int
    selected_count: int


class QueryResponse(BaseModel):
    """Response body for the ``POST /chat/query`` endpoint.

    Attributes:
        answer (str): The LLM-generated answer to the user's question.
        sources (list[SourceResponse]): The subset of retrieved chunks used
            as context for the answer.
        process_time_ms (float): End-to-end request processing time in
            milliseconds, measured by the route handler.
        debug (QueryDebugResponse): Debug metadata and metrics from graph execution.
    """

    answer: str
    sources: list[SourceResponse]
    process_time_ms: float
    debug: QueryDebugResponse



class DocumentResponse(BaseModel):
    """Serialised representation of a single document record.

    Used as an element of :attr:`DocumentsListResponse.documents`.

    Attributes:
        id (int): The document's primary key.
        file_name (str): The stored filename of the PDF.
        created_at (str): ISO-format timestamp of when the document was ingested.
    """

    id: int
    file_name: str
    created_at: str


class DocumentsListResponse(BaseModel):
    """Response body for the ``GET /documents`` endpoint.

    Attributes:
        documents (list[DocumentResponse]): All ingested documents, ordered
            by creation time descending.
    """

    documents: list[DocumentResponse]


class DeleteDocumentResponse(BaseModel):
    """Response body for the ``DELETE /documents/{DOCUMENT_ID}`` endpoint.

    Attributes:
        message (str): Human-readable confirmation of the deletion.
        DOCUMENT_ID (int): The ID of the document that was deleted.
    """

    message: str
    DOCUMENT_ID: int


