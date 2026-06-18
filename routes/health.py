from fastapi import APIRouter

from schemas import HealthResponse

router = APIRouter(
    tags=["Health"]
    )

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        message="PDF RAG Chat API is running!"
    )
