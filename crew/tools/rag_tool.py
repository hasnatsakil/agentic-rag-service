"""
Custom CrewAI tool — PDF document search backed by the RAG graph.

This module defines ``search_pdf``, a CrewAI-compatible tool that wraps
:meth:`~services.graph_services.GraphService.ask_pdf_with_graph`.  When a
CrewAI agent calls this tool, it runs the full LangGraph agentic pipeline
(hybrid search, grading, re-ranking, answer generation) and returns the
selected source chunks as plain text for the agent to reason over.

Note:
    This tool is only used by the CLI-based CrewAI crew
    (``crew/crews/rag_crew.py``).  It is **not** part of the FastAPI
    web service.
"""

import sys
import os

# Add project root to sys.path so imports resolve regardless of working directory.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from crewai.tools import tool
from services.graph_services import GraphService
from config import settings


@tool("Search PDF Documents")
def search_pdf(query: str) -> str:
    """Search ingested PDF documents in the vector database for relevant information.

    Delegates to :meth:`~services.graph_services.GraphService.ask_pdf_with_graph`,
    which runs the full agentic RAG pipeline (hybrid retrieval, relevance
    grading, context selection).  Returns the source chunks selected as context
    rather than the generated answer, so the calling CrewAI agent can reason
    directly over the raw evidence.

    Use this tool whenever you need to find factual information from the
    uploaded PDF documents.  The input must be a clear, specific search
    query string.

    Args:
        query: A natural-language search query describing the information to
            retrieve (e.g. ``"visa requirements for international students"``).

    Returns:
        A plain-text string containing numbered source chunks from the
        documents, or a fallback message if no relevant information was found.

    Example::

        result = search_pdf("What is the tuition fee for international students?")
        # -> "Chunk 1:\\nThe tuition fee for international students is...\\n\\n..."
    """
    result = GraphService.ask_pdf_with_graph(
        question=query,
        SEARCH_K=settings.SEARCH_K,
        ANSWER_K=settings.ANSWER_K,
        MIN_SCORE=settings.MIN_SCORE,
        MAX_CONTEXT_CHARS=settings.MAX_CONTEXT_CHARS,
    )

    if not result.sources:
        return "No relevant information found in the PDF documents."

    # Format each selected chunk with a sequential label for the agent.
    chunks = ""
    for i, source in enumerate(result.sources, 1):
        chunks += f"Chunk {i}:\n{source.chunk_text}\n\n"

    return chunks