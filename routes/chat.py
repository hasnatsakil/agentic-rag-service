import time
from fastapi import APIRouter, Depends, HTTPException

from dependencies import get_vector_store
from schemas import QueryRequest, QueryResponse
from services.graph_services import GraphService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
    )

def format_query_response(
        result,
        process_time_ms: float
    ):
    return {
        "answer": result.answer,
        "sources": [
            {
                "label" : source.label(),
                "score" : source.score,
                "chunk_id" : source.chunk_id,
                "page_number" : source.page_number,
                "chunk_text" : source.chunk_text
            }
            for source in result.sources
        ],
        "process_time_ms": process_time_ms
    }

    
@router.post(
        "/query", 
        response_model=QueryResponse,
        summary="Query a Document via LangGraph",
        description="Passes the user's question through a stateful graph that retrieves vectors, "
        "dynamically rewrites queries if needed, and conditionally generates an LLM response."
        )
def query_pdf_graph(
    request: QueryRequest,
    vector_store = Depends(get_vector_store)
    ):
    start_time = time.time()
    if not request.question.strip():
        raise HTTPException(
            status_code = 400,
            detail = "Question cannot be empty."
        )
    try:
        documents = vector_store.list_documents()
        doc_ids = {doc["id"] for doc in documents}

        if request.DOCUMENT_ID not in doc_ids:
            raise HTTPException(
                status_code = 400,
                detail = f"Document not found, DOCUMENT_ID:{request.DOCUMENT_ID}",
            )
        result = GraphService.ask_pdf_with_graph(
            question = request.question,
            DOCUMENT_ID = request.DOCUMENT_ID,
            SEARCH_K = request.SEARCH_K,
            ANSWER_K = request.ANSWER_K,
            MIN_SCORE = request.MIN_SCORE,
            MAX_CONTEXT_CHARS = request.MAX_CONTEXT_CHARS
        )
        end_time = time.time()
        process_time_ms = round((end_time - start_time) * 1000, 2)
        return format_query_response(result, process_time_ms)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to process query: {str(e)}"
        )
    
