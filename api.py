from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse
import logging

from routes.health import router as health_router
from routes.documents import router as documents_router
from routes.chat import router as chat_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF RAG Chat API",
    description = "Chat with PDF/Documents using Openrouter + Neon pgvector",
    version = "0.4.0"
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "/docs",
        "health": "/health",
        "documents": "/documents",
        "chat": "/chat"
    }

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
    ):
    logger.error(f"Unhandled Crash on {request.url.path}: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )
