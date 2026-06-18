from pydantic import BaseModel, Field

from config import settings

class QueryRequest(BaseModel):
    document_id: int = Field(
        ..., 
        description="The ID of the document to query"
        )
    question: str = Field(
        ...,
        description="Question to ask about the document"
        )
    search_k: int = Field(
        settings.search_k,
        description="Number of chunks to retrieve from vector store for context"
        )
    answer_k: int = Field(
        settings.answer_k,
        description="Number of chunks sent to LLM"
        )
    min_score: float = Field(
        settings.min_score,
        description="Minimum cosine similarity score"
        )
    max_context_chars: int = Field(
        settings.max_context_chars,
        description="Maximum total characters of context to send to LLM"
    )

class HealthResponse(BaseModel):
    status: str
    message: str

class UploadResponse(BaseModel):
    message: str
    document_id: int
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

class DocumentResponse(BaseModel):
    id: int
    file_name: str
    created_at: str

class DocumentsListResponse(BaseModel):
    documents: list[DocumentResponse]

class DeleteDocumentResponse(BaseModel):
    message: str
    document_id: int
