"""
Graph-based RAG orchestration service.

This module provides :class:`GraphService`, a thin facade over the compiled
LangGraph RAG workflow (:data:`~graph.rag_graph.rag_graph`).  It translates
the flat function-call interface used by the FastAPI chat route into the
typed state dictionary required by the graph, invokes the graph, and returns
a :class:`~core.models.ChatResult`.

The underlying graph implements a self-correcting RAG loop:
``agent → execute_tool → grade_documents → (select_context | rewrite_query |
no_context) → answer → check_hallucination``.
"""


from core.models import ChatResult
from graph.rag_graph import rag_graph
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphService:
    """Facade for invoking the stateful LangGraph RAG pipeline.

    Wraps :data:`~graph.rag_graph.rag_graph` with a clean, typed method
    signature and handles state initialisation and result extraction.
    """

    @staticmethod
    def ask_pdf_with_graph(
        question: str,
        DOCUMENT_ID: int,
        SEARCH_K: int = settings.SEARCH_K,
        GRADE_K:int = settings.GRADE_K,
        ANSWER_K: int = settings.ANSWER_K,
        MIN_SCORE: float = settings.MIN_SCORE,
        MAX_CONTEXT_CHARS: int = settings.MAX_CONTEXT_CHARS,
        use_llm_rerank: bool = False,
    ) -> ChatResult:
        """Run the full RAG graph pipeline for a user question.

        Initialises the :class:`~graph.rag_graph.RAGState` with the provided
        parameters and default values for transient graph fields, invokes
        the compiled graph, logs the outcome, and returns a
        :class:`~core.models.ChatResult`.

        Args:
            question: The user's natural-language question.
            DOCUMENT_ID: ID of the document to query, as returned by
                :meth:`~core.vector_store.NeonVectorStore.create_document`.
            SEARCH_K: Number of candidate chunks to retrieve from the vector
                store before grading. Defaults to ``settings.SEARCH_K``.
            ANSWER_K: Maximum number of chunks forwarded to the LLM for
                answer generation after grading and context selection.
                Defaults to ``settings.ANSWER_K``.
            MIN_SCORE: Minimum retrieval score threshold.  Chunks scoring
                below this value are discarded after retrieval.
                Defaults to ``settings.MIN_SCORE``.
            MAX_CONTEXT_CHARS: Hard cap on total context characters sent to
                the LLM.  Prevents prompt overflow. Defaults to
                ``settings.MAX_CONTEXT_CHARS``.

        Returns:
            A :class:`~core.models.ChatResult` containing:

            - ``answer`` — the LLM-generated response string.
            - ``sources`` — the list of :class:`~core.models.RetrievalResult`
              objects used as context.

        Example::

            result = GraphService.ask_pdf_with_graph(
                question="What was the revenue in 2023?",
                DOCUMENT_ID=5,
                SEARCH_K=8,
                ANSWER_K=3,
            )
            print(result.answer)
        """
        result = rag_graph.invoke(
            {
                "question": question,
                "search_question": question,
                "DOCUMENT_ID": DOCUMENT_ID,
                "SEARCH_K": SEARCH_K,
                "GRADE_K": GRADE_K,
                "ANSWER_K": ANSWER_K,
                "MIN_SCORE": MIN_SCORE,
                "MAX_CONTEXT_CHARS": MAX_CONTEXT_CHARS,
                # Transient state fields — initialised to empty defaults.
                "chunks": "",
                "answer": "",
                "filtered_results": [],
                "selected_results": [],
                "has_context": False,
                "retry_count": 0,
                "max_retries": 1,
                "used_rewrite": False,
                "is_grounded": False,
                "use_llm_rerank": use_llm_rerank,
            }
        )

        logger.info(
            f"Graph completed: used_rewrite={result['used_rewrite']}, "
            f"sources={len(result['selected_results'])}"
        )

        return ChatResult(
            answer=result["answer"],
            sources=result["selected_results"],
            used_rewrite=result.get("used_rewrite", False),
            is_grounded= result.get("is_grounded", False),
            retrieval_count= len(result.get("filtered_results", [])),
            selected_count= len(result.get("selected_results", [])),
        )
