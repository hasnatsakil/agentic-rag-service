from fastapi import APIRouter, Depends
from dependencies import get_vector_store
from core.vector_store import NeonVectorStore

router = APIRouter(
    tags=["Health"]
    )

@router.get("/health")
def health(
    vector_store: NeonVectorStore = Depends(get_vector_store)
    ):
    try:
        # Ping the DB (A very light query just to ensure connection)
        vector_store.list_documents()
        return {
            "status": "ok",
            "database": "connected",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": f"disconnected", 
            "error": {str(e)}
        }
