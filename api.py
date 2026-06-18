from fastapi import FastAPI

from routes.health import router as health_router
from routes.documents import router as documents_router
from routes.chat import router as chat_router


app = FastAPI(
    title="PDF RAG Chat API",
    description = "Chat with PDF/Documents using Openrouter + Neon pgvector",
    version = "0.4.0"
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "/docs",
        "health": "/health",
        "documents": "/documents",
        "chat": "/chat"
    }

