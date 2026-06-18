from ingest_service import IngestService
from chat_service import ChatService
from vector_store import NeonVectorStore

def get_ingest_service() -> IngestService:
    return IngestService()

def get_chat_service() -> ChatService:
    return ChatService()

def get_vector_store() -> NeonVectorStore:
    return NeonVectorStore()