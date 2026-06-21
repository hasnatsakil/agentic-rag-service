from core.embeddings import OpenRouterEmbeddingClient
from core.rag_engine import RAGEngine
from core.vector_store import NeonVectorStore
from core.models import ChatResult

class ChatService:
    @staticmethod
    def ask_pdf(
        question: str,
        document_id: int,
        search_k: int = 8,
        answer_k: int = 3,
        min_score: float = 0.3,
        max_context_chars: int = 3000,
        ) -> ChatResult:
        query_embedding = OpenRouterEmbeddingClient.embed_query(question)

        results = NeonVectorStore.similarity_search(
            query_embedding = query_embedding,
            top_k = search_k,
            document_id = document_id,
        )
        RAGEngine.print_retrieval_debug(results, min_score=min_score)
        filtered_results = [
            result for result in results
            if result.score >= min_score
        ]
        if not filtered_results:
            return ChatResult(
                answer=(
                    "I could not find enough relevant context in this document to answer that. "
                    f"Try lowering min_score below {min_score} or asking a more specific question."
                ),
                sources=[],
            )
        selected_results = []
        current_chars = 0

        for result in filtered_results:
            next_size = len(result.chunk_text)

            if current_chars + next_size > max_context_chars:
                break
            selected_results.append(result)
            current_chars += next_size

            if len(selected_results) >= answer_k:
                break
        if not selected_results:
            return ChatResult(
                answer=(
                    "Relevant chunks were found, but they were too large to fit within "
                    f"the max_context_chars limit of {max_context_chars}."
                ),
                sources=[],
            )

        context = RAGEngine.build_context(selected_results)

        answer = RAGEngine.generate_answer(
            question = question,
            context = context
            )
        return ChatResult(
            answer=answer, 
            sources=selected_results)