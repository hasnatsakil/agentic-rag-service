"""
Application settings for the PDF RAG Chat API.

This module exposes a singleton :data:`settings` object with sensible
defaults for all tunable RAG parameters.  Values can be overridden at
the API call level via the :class:`~schemas.QueryRequest` body.

Attributes of :class:`Settings`:
    SEARCH_K (int): Number of candidate chunks retrieved from the vector
        store before grading. Higher values improve recall at the cost of
        more LLM grading calls.
    ANSWER_K (int): Maximum chunks forwarded to the LLM after grading and
        context selection. Controls prompt size.
    MIN_SCORE (float): Minimum retrieval score threshold. Chunks below this
        value are discarded. ``0.0`` disables filtering.
    MAX_CONTEXT_CHARS (int): Hard cap on total context characters sent to the
        LLM. Prevents prompt overflow.
    DOCUMENT_ID (int): Default document ID used when no explicit ID is
        provided. Useful for local development and testing.
"""


class Settings:
    """Centralised configuration constants for the RAG pipeline.

    All attributes are class-level constants and can be read directly
    without instantiation (e.g. ``Settings.SEARCH_K``), or accessed via
    the module-level :data:`settings` singleton.
    """

    #: Candidate chunks to retrieve per query before grading.
    SEARCH_K: int = 15

    GRADE_K = 6

    #: Top chunks forwarded to the LLM for answer generation.
    ANSWER_K: int = 3

    #: Minimum score threshold; chunks below this are filtered out.
    MIN_SCORE: float = 0.0

    #: Maximum context characters sent to the LLM per request.
    MAX_CONTEXT_CHARS: int = 3000

    #: Default document ID for local development and testing.
    DOCUMENT_ID: int = 2


#: Module-level singleton — import and use this everywhere.
settings = Settings()
