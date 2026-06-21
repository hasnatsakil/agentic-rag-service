from fastapi import APIRouter, Depends, HTTPException

from services.chat_service import ChatService
from dependencies import get_chat_service, get_vector_store
from schemas import QueryRequest, QueryResponse

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
    )

@router.post("/query", response_model=QueryResponse)
def query_pdf(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
    vector_store = Depends(get_vector_store)
    ):
    if not request.question.strip():
        raise HTTPException(
            status_code = 400,
            detail = "Question cannot be empty."
        )
    try:
        documents = vector_store.list_documents()
        doc_ids = {doc["id"] for doc in documents}

        if request.document_id not in doc_ids:
            raise HTTPException(
                status_code = 400,
                detail = f"Document not found, document_id:{request.document_id}",
            )
        result = chat_service.ask_pdf(
            question = request.question,
            document_id = request.document_id,
            search_k = request.search_k,
            answer_k = request.answer_k,
            min_score = request.min_score,
            max_context_chars = request.max_context_chars
        )
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
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to process query: {str(e)}"
        )