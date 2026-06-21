from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from services.ingest_service import IngestService
from dependencies import get_ingest_service, get_vector_store
from schemas import (
    UploadResponse,
    DocumentsListResponse,
    DeleteDocumentResponse
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    ingest_service: IngestService = Depends(get_ingest_service),
    ):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Upload file must have a filename."
        )
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed right now."
        )
    
    # Some FastAPI/Starlette versions may not always provide file.size.
    # So we use a safer file-size check.
    file.file.seek(0, 2)  # Move to end of file
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to start of file

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )
    
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        result = ingest_service.ingest_pdf(
            temp_path,
            original_file_name=file.filename
        )
        return {
            "message": "File Uploaded and Ingested Successfully",
            "document_id": result.document_id,
            "query_hint": f"Use document_id {result.document_id} to query",
            "file_name": result.file_name,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and ingest PDF: {str(e)}"
        )
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

@router.get("", response_model=DocumentsListResponse)
def list_documents(
    vector_store = Depends(get_vector_store),
    ):
    try:
        documents = vector_store.list_documents()
        return {
            "documents": [
                {
                    "id": doc["id"],
                    "file_name": doc["file_name"],
                    "created_at": doc["created_at"]
                }
                for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )
    
@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: int,
    vector_store = Depends(get_vector_store),
    ):
    try:
        documents = vector_store.list_documents()
        doc_ids = {doc["id"] for doc in documents}

        if document_id not in doc_ids:
            raise HTTPException(
                status_code=404,
                detail = f"Document not found with id {document_id}."
            )
        vector_store.delete_document(document_id)

        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )