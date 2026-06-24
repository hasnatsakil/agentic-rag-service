from core.models import ChatResult
from graph.rag_graph import rag_graph
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphService:
    
    @staticmethod
    def ask_pdf_with_graph(
            question: str,
            DOCUMENT_ID: int,
            SEARCH_K: int = settings.SEARCH_K,
            ANSWER_K: int = settings.ANSWER_K,
            MIN_SCORE: float = settings.MIN_SCORE,
            MAX_CONTEXT_CHARS: int = settings.MAX_CONTEXT_CHARS
        ) -> ChatResult:
        result = rag_graph.invoke(
            {
                "question": question,
                "search_question": question,
                "DOCUMENT_ID": DOCUMENT_ID,
                "SEARCH_K": SEARCH_K,
                "ANSWER_K": ANSWER_K,
                "MIN_SCORE": MIN_SCORE,
                "MAX_CONTEXT_CHARS": MAX_CONTEXT_CHARS,
                "chunks": "",
                "answer": "",
                "filtered_results": [],
                "selected_results": [],
                "has_context": False,
                "retry_count": 0,
                "max_retries": 1,
                "used_rewrite": False
            }
        )
        logger.info(
            f"Graph completed: used_rewrite={result['used_rewrite']}, sources={len(result['selected_results'])}"
        )

        return ChatResult(
            answer = result["answer"],
            sources = result["selected_results"]
        )
