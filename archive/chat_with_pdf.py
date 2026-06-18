
from embeddings import OpenRouterEmbeddingClient
from rag_engine import RAGEngine
from vector_store import NeonVectorStore

def chat_with_pdf(
    question: str,
    document_id: int,
    top_k: int = 3,
    ) -> str:
    query_embedding = OpenRouterEmbeddingClient.embed_query(question)

    result = NeonVectorStore.similarity_search(
        query_embedding = query_embedding,
        top_k = top_k,
        document_id = document_id,
    )

    RAGEngine.print_retrieval_debug(result)

    context = RAGEngine.build_context(result)

    answer = RAGEngine.generate_answer(
        question = question,
        context = context
        )
    print("Answer:", answer)
    return answer

if __name__ == "__main__":
    document_id = 5
    question = "What is the document about?"

    chat_with_pdf(
        question = question,
        document_id = document_id,
        top_k = 3
    )