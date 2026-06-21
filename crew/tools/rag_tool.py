import sys, os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from crewai.tools import tool
from services.chat_service import ChatService

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
    document_id = 1

    result = ChatService().ask_pdf(
        question = query,
        document_id = document_id,
        search_k = 8,
        answer_k = 5,
        min_score = 0.3,
        max_context_chars = 4000
    )

    if not result.sources:
        return "No relevant information found in the PDF documents."
    
    chunks = ""

    for i, source in enumerate(result.sources, 1):
        chunks += f"Chunk {i}:\n{source.chunk_text}\n\n"
    
    return chunks