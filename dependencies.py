from services.ingest_service import IngestService
from services.graph_services import GraphService
from core.vector_store import NeonVectorStore

def get_ingest_service() -> IngestService:
    return IngestService()

def get_graph_service() -> GraphService:
    return GraphService()

def get_vector_store() -> NeonVectorStore:
    return NeonVectorStore()