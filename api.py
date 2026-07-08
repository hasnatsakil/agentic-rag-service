"""
FastAPI application entry point for the PDF RAG Chat API.

This module creates and configures the :class:`~fastapi.FastAPI` application
instance, registers all API routers, and installs a global exception handler
to ensure that unhandled server errors are logged and returned as clean
JSON responses rather than stack traces.

Routers registered:
    - :mod:`routes.health`     — ``GET /health``
    - :mod:`routes.documents`  — ``POST/GET/DELETE /documents``
    - :mod:`routes.chat`       — ``POST /chat/query``

Running locally::

    uvicorn api:app --reload
"""

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse
import logging

from routes.health import router as health_router
from routes.documents import router as documents_router
from routes.chat import router as chat_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF RAG Chat API",
    description="Chat with PDF/Documents using Openrouter + Neon pgvector",
    version="0.4.0",
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.head("/")
def head_root():
    """Respond to HEAD requests at the root path for uptime monitoring."""
    return Response(status_code=200)


@app.get("/")
def root():
    """Return a directory of available API endpoints.

    Returns:
        A JSON object mapping endpoint names to their paths.
    """
    return {
        "status": "ok",
        "message": "/docs",
        "health": "/health",
        "documents": "/documents",
        "chat": "/chat",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unhandled exceptions.

    Logs the full traceback server-side and returns a generic 500 JSON
    response to the client, preventing internal error details from leaking.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc: The unhandled exception instance.

    Returns:
        A :class:`~fastapi.responses.JSONResponse` with status ``500`` and
        a ``{"detail": "Internal Server Error"}`` body.
    """
    logger.error(
        f"Unhandled Crash on {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
