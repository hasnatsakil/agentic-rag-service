import math

from core.embeddings import OpenRouterEmbeddingClient
from core.models import RetrievalResult
from services.agent_completion import agent_complete

class RAGEngine:
    TOP_K: int = 2

    @classmethod
    def embed_chunks(
            cls,
            chunks: list[str]
            ):
        embeddings = OpenRouterEmbeddingClient.embed_documents(chunks)
        return embeddings 

    @classmethod
    def cosine_similarity(
        cls,
        vector_a: list[float], 
        vector_b: list[float]
        ) -> float:
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        length_a = math.sqrt(sum(a * a for a in vector_a))
        length_b = math.sqrt(sum(b * b for b in vector_b))

        if length_a == 0 or length_b == 0:
            return 0

        return dot_product / (length_a * length_b)

    @classmethod
    def retrieve(
        cls, 
        question: str, 
        chunks: list[str], 
        embeddings: list[list[float]], 
        top_k: int = TOP_K
        ) -> list[RetrievalResult]:
        question_embedding = OpenRouterEmbeddingClient.embed_query(question)
        results = []

        for i, chunk in enumerate(chunks):
            score = cls.cosine_similarity(question_embedding, embeddings[i])
            results.append(
                RetrievalResult(
                    score=score,
                    chunk_id=i,
                    chunk_text=chunk
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
    
    @classmethod
    def build_context(
        cls, 
        results: list[RetrievalResult]
        ) -> str:
        context_parts = []
        for result in results:
            label = f"[{result.label()}]"

            context_parts.append(f"{label}\n{result.chunk_text}")
        context = "\n\n".join(context_parts)
        
        return context

    @classmethod
    def generate_answer(
            cls,
            question: str,
            context: str
            ) -> str:
        if not context.strip():
            return "I could not find enough context to answer that."
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict PDF question-answering assistant. "
                    "Answer using only the provided context. "
                    "Do not use outside knowledge. "
                    "If the context does not contain the answer, say you cannot find it in the document. "
                    "Mention the chunk label you used, like [Page 1, Chunk 2]. "
                    "Keep the answer short."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer only from the context above.",
            }
        ]

        # 6. Call the LLM completion
        completion = agent_complete(messages)
        answer = completion.choices[0].message.content
        return answer

    @classmethod
    def print_retrieval_debug(
        cls, 
        top_results : list[RetrievalResult],
        MIN_SCORE: float | None = None,
        ) -> None:
        threshold_label = f" (threshold: {MIN_SCORE})" if MIN_SCORE is not None else ""
        print(f"\n── Retrieval Debug{threshold_label} ──────────────────────")

        for result in top_results:
            if MIN_SCORE is not None:
                status = "✓ PASS" if result.score >= MIN_SCORE else "✗ FAIL"
                print(f"  [{status}] {result.label()}, Score: {result.score:.4f}")
            else:
                print(f"  {result.label()}, Score: {result.score:.4f}")
            snippet = result.chunk_text[:120].replace("\n", " ")
            print(f"           {snippet}…")
            print()
        print("────────────────────────────────────────────────────")