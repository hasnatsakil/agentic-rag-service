import sys, os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from crewai.tools import tool
from services.graph_services import GraphService
from config import settings

@tool("Search PDF Documents")
def search_pdf(
    query: str
    ) -> str:
    """
    Searches the ingested PDF documents in the vector database for relevent
    Information. 
    Use this tool whenever you need to find information from the PDF documents.
    Input must be a clear search query string.
    Returns relevant text chunks from the documents.
    """
    
    
    result = GraphService.ask_pdf_with_graph(
        question = query,
        DOCUMENT_ID = settings.DOCUMENT_ID,
        SEARCH_K = settings.SEARCH_K,
        ANSWER_K = settings.ANSWER_K,
        MIN_SCORE = settings.MIN_SCORE,
        MAX_CONTEXT_CHARS = settings.MAX_CONTEXT_CHARS
    )

    if not result.sources:
        return "No relevant information found in the PDF documents."
    
    chunks = ""

    for i, source in enumerate(result.sources, 1):
        chunks += f"Chunk {i}:\n{source.chunk_text}\n\n"
    
    return chunks