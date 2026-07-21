"""
Chat route — ``POST /chat/query`` and session management endpoints.

Handles incoming question requests, fetches conversation history and summaries,
delegates the full RAG pipeline to :class:`~services.graph_services.GraphService`,
queues asynchronous history persistence & summary compaction tasks,
and provides session lifecycle management endpoints.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from core.chat_history_store import ChatHistoryStore
from services.compaction_service import save_and_compact_workflow
from dependencies import get_vector_store
from schemas import QueryRequest, QueryResponse
from services.graph_services import GraphService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def format_query_response(result, process_time_ms: float) -> dict:
    """Serialise a :class:`~core.models.ChatResult` into a response dict matching QueryResponse.

    Args:
        result: A :class:`~core.models.ChatResult` returned by the graph.
        process_time_ms: End-to-end request duration in milliseconds.

    Returns:
        A dict matching :class:`~schemas.QueryResponse` JSON structure.
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
        "debug": {
            "used_rewrite": result.used_rewrite,
            "is_grounded": result.is_grounded,
            "retrieval_count": result.retrieval_count,
            "selected_count": result.selected_count,
        }
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query a Document via LangGraph",
    description=(
        "Passes the user's question through a stateful graph that retrieves vectors, "
        "dynamically rewrites queries if needed, grades context relevance, and generates an LLM response."
    ),
)
def query_pdf_graph(
    request: QueryRequest,
    background_task: BackgroundTasks,
    vector_store=Depends(get_vector_store)
):
    """Handle a document Q&A request with session memory.

    1. Validates that the question is non-empty and documents exist.
    2. Fetches recent chat history and running summary for `request.session_id`.
    3. Delegates execution to :meth:`GraphService.ask_pdf_with_graph`.
    4. Enqueues background task to save turn and update running summary.

    Args:
        request: Validated :class:`~schemas.QueryRequest` body.
        background_task: Injected FastAPI :class:`~fastapi.BackgroundTasks`.
        vector_store: Injected :class:`~core.vector_store.NeonVectorStore`.

    Returns:
        A :class:`~schemas.QueryResponse` with answer, sources, debug metrics, and timing.

    Raises:
        HTTPException (400): If the question is blank or no documents exist.
        HTTPException (500): If an internal processing error occurs.
    """
    start_time = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        documents = vector_store.list_documents()

        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No documents found in the database. Please upload a PDF first.",
            )
    
        history = ChatHistoryStore.get_last_20_message(request.session_id)
        summary = ChatHistoryStore.get_summary(request.session_id)

        result = GraphService.ask_pdf_with_graph(
            question=request.question,
            SEARCH_K=request.SEARCH_K,
            GRADE_K=request.GRADE_K,
            ANSWER_K=request.ANSWER_K,
            MIN_SCORE=request.MIN_SCORE,
            MAX_CONTEXT_CHARS=request.MAX_CONTEXT_CHARS,
            use_llm_rerank=request.use_llm_rerank,
            available_documents=documents,
            history=history,
            summary=summary
        )
        background_task.add_task(
            save_and_compact_workflow,
            session_id=request.session_id,
            user_question=request.question,
            assistant_answer=result.answer,
            old_summary=summary
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

# ------------------------------------------------------------------ #
#  Session Lifecycle Endpoints                                       #
# ------------------------------------------------------------------ #

@router.get("/sessions")
def list_chat_sessions(vector_store=Depends(get_vector_store)):
    """Retrieve all active chat thread session IDs with their latest activity timestamps."""
    try:
        return {"sessions": ChatHistoryStore.list_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Retrieve the full chronological message log of a specific chat session."""
    try:
        return {"messages": ChatHistoryStore.get_all_session_messages(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str):
    """Clear all chat history messages and running summaries for a specific session ID."""
    try:
        ChatHistoryStore.delete_session(session_id)
        return {"message": "Session deleted successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

