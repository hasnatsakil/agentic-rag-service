"""
Chat route — ``POST /chat/query``.

Handles incoming question requests, validates the target document exists,
delegates the full RAG pipeline to :class:`~services.graph_services.GraphService`,
and returns the answer with its source citations and processing time.
"""

import time
from fastapi import APIRouter, Depends, HTTPException

from dependencies import get_vector_store
from schemas import QueryRequest, QueryResponse
from services.graph_services import GraphService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def format_query_response(result, process_time_ms: float) -> dict:
    """Serialise a :class:`~core.models.ChatResult` into a response dict.

    Converts each source :class:`~core.models.RetrievalResult` to a plain
    dict that matches the :class:`~schemas.SourceResponse` schema, and
    attaches the measured processing time.

    Args:
        result: A :class:`~core.models.ChatResult` returned by the graph.
        process_time_ms: End-to-end request duration in milliseconds.

    Returns:
        A dict suitable for FastAPI to validate against
        :class:`~schemas.QueryResponse` and serialise to JSON.
    """
    return {
        "answer": result.answer,
        "sources": [
            {
                "label": source.label(),
                "score": source.score,
                "chunk_id": source.chunk_id,
                "page_number": source.page_number,
                "chunk_text": source.chunk_text,
            }
            for source in result.sources
        ],
        "process_time_ms": process_time_ms,
        "debug":{
            "used_rewrite":result.used_rewrite,
            "is_grounded":result.is_grounded,
            "retrieval_count":result.retrieval_count,
            "selected_count":result.selected_count,
        }
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query a Document via LangGraph",
    description=(
        "Passes the user's question through a stateful graph that retrieves vectors, "
        "dynamically rewrites queries if needed, and conditionally generates an LLM response."
    ),
)
def query_pdf_graph(
    request: QueryRequest,
    vector_store=Depends(get_vector_store),
):
    """Handle a document Q&A request.

    Validates that the question is non-empty and that the requested document
    exists in the store, then delegates to :meth:`GraphService.ask_pdf_with_graph`
    for the full self-correcting RAG pipeline.

    Args:
        request: Validated :class:`~schemas.QueryRequest` body containing
            the question and retrieval parameters.
        vector_store: Injected :class:`~core.vector_store.NeonVectorStore`
            instance (via :func:`~dependencies.get_vector_store`).

    Returns:
        A :class:`~schemas.QueryResponse` with the answer, source citations,
        and processing time.

    Raises:
        HTTPException (400): If the question is blank or the document ID
            does not exist.
        HTTPException (500): If an unexpected error occurs during processing.
    """
    start_time = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        documents = vector_store.list_documents()
        doc_ids = {doc["id"] for doc in documents}

        if request.DOCUMENT_ID not in doc_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Document not found, DOCUMENT_ID:{request.DOCUMENT_ID}",
            )

        result = GraphService.ask_pdf_with_graph(
            question=request.question,
            DOCUMENT_ID=request.DOCUMENT_ID,
            SEARCH_K=request.SEARCH_K,
            GRADE_K = request.GRADE_K,
            ANSWER_K=request.ANSWER_K,
            MIN_SCORE=request.MIN_SCORE,
            MAX_CONTEXT_CHARS=request.MAX_CONTEXT_CHARS,
            use_llm_rerank=request.use_llm_rerank
        )

        process_time_ms = round((time.time() - start_time) * 1000, 2)
        return format_query_response(result, process_time_ms)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}",
        )
