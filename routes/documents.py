"""
Documents route — ``/documents`` CRUD endpoints.

Handles PDF upload and ingestion, document listing, and document deletion.
All routes validate inputs early and delegate business logic to the
injected service/store dependencies.

Endpoints:
    - ``POST   /documents/upload``       — upload and ingest a PDF file
    - ``GET    /documents``              — list all ingested documents
    - ``DELETE /documents/{DOCUMENT_ID}``— delete a document and its chunks
"""

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from services.ingest_service import IngestService
from dependencies import get_ingest_service, get_vector_store
from schemas import (
    UploadResponse,
    DocumentsListResponse,
    DeleteDocumentResponse,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

#: Maximum permitted upload size (10 MB).
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    ingest_service: IngestService = Depends(get_ingest_service),
):
    """Upload a PDF file and ingest it into the vector store.

    Validates the file (name, extension, size, non-empty), writes it to a
    temporary file on disk, and delegates ingestion to
    :meth:`~services.ingest_service.IngestService.ingest_pdf`.  The
    temporary file is cleaned up in a ``finally`` block regardless of
    success or failure.

    Args:
        file: The uploaded file, must be a non-empty ``.pdf`` ≤ 10 MB.
        ingest_service: Injected :class:`~services.ingest_service.IngestService`
            (via :func:`~dependencies.get_ingest_service`).

    Returns:
        An :class:`~schemas.UploadResponse` with the new document's ID,
        filename, page count, and chunk count.

    Raises:
        HTTPException (400): If the file has no name, is not a PDF, or is empty.
        HTTPException (413): If the file exceeds the 10 MB size limit.
        HTTPException (500): If ingestion fails for any other reason.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload file must have a filename.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed right now.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum file size is {MAX_FILE_SIZE / (1024 * 1024)} MB.",
        )

    # Some FastAPI/Starlette versions may not always provide file.size,
    # so we use a safer seek-based file-size check.
    file.file.seek(0, 2)  # Move to end of file.
    file_size = file.file.tell()
    file.file.seek(0)     # Reset to start of file.

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        result = ingest_service.ingest_pdf(temp_path, original_file_name=file.filename)

        return {
            "message": "File Uploaded and Ingested Successfully",
            "DOCUMENT_ID": result.DOCUMENT_ID,
            "query_hint": f"Use DOCUMENT_ID {result.DOCUMENT_ID} to query",
            "file_name": result.file_name,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and ingest PDF: {str(e)}",
        )
    finally:
        # Always clean up the temporary file, even on error.
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@router.get("", response_model=DocumentsListResponse)
def list_documents(
    vector_store=Depends(get_vector_store),
):
    """List all ingested documents.

    Args:
        vector_store: Injected :class:`~core.vector_store.NeonVectorStore`
            (via :func:`~dependencies.get_vector_store`).

    Returns:
        A :class:`~schemas.DocumentsListResponse` containing all documents
        ordered by creation time descending.

    Raises:
        HTTPException (500): If the database query fails.
    """
    try:
        documents = vector_store.list_documents()
        return {
            "documents": [
                {
                    "id": doc["id"],
                    "file_name": doc["file_name"],
                    "created_at": doc["created_at"],
                }
                for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/{DOCUMENT_ID}", response_model=DeleteDocumentResponse)
def delete_document(
    DOCUMENT_ID: int,
    vector_store=Depends(get_vector_store),
):
    """Delete a document and all its associated chunks.

    Verifies that the document exists before attempting deletion so that a
    404 is returned instead of a silent no-op for unknown IDs.

    Args:
        DOCUMENT_ID: The primary key of the document to delete, taken from
            the URL path.
        vector_store: Injected :class:`~core.vector_store.NeonVectorStore`
            (via :func:`~dependencies.get_vector_store`).

    Returns:
        A :class:`~schemas.DeleteDocumentResponse` confirming the deletion.

    Raises:
        HTTPException (404): If no document with ``DOCUMENT_ID`` exists.
        HTTPException (500): If the deletion query fails.
    """
    try:
        documents = vector_store.list_documents()
        doc_ids = {doc["id"] for doc in documents}

        if DOCUMENT_ID not in doc_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found with id {DOCUMENT_ID}.",
            )

        vector_store.delete_document(DOCUMENT_ID)

        return {
            "message": "Document deleted successfully",
            "DOCUMENT_ID": DOCUMENT_ID,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")