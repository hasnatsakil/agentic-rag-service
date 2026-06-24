from pydantic import BaseModel, Field

from config import settings

class QueryRequest(BaseModel):
    DOCUMENT_ID: int = Field(
        ..., 
        description="The ID of the document to query"
        )
    question: str = Field(
        ...,
        description="Question to ask about the document"
        )
    SEARCH_K: int = Field(
        settings.SEARCH_K,
        description="Number of chunks to retrieve from vector store for context"
        )
    ANSWER_K: int = Field(
        settings.ANSWER_K,
        description="Number of chunks sent to LLM"
        )
    MIN_SCORE: float = Field(
        settings.MIN_SCORE,
        description="Minimum cosine similarity score"
        )
    MAX_CONTEXT_CHARS: int = Field(
        settings.MAX_CONTEXT_CHARS,
        description="Maximum total characters of context to send to LLM"
    )

class HealthResponse(BaseModel):
    status: str
    message: str

class UploadResponse(BaseModel):
    message: str
    DOCUMENT_ID: int
    file_name: str
    page_count: int
    chunk_count: int

class SourceResponse(BaseModel):
    label: str
    score: float
    chunk_id: int
    page_number: int | None
    chunk_text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    process_time_ms:float

class DocumentResponse(BaseModel):
    id: int
    file_name: str
    created_at: str

class DocumentsListResponse(BaseModel):
    documents: list[DocumentResponse]

class DeleteDocumentResponse(BaseModel):
    message: str
    DOCUMENT_ID: int
