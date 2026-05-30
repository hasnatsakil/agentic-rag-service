from embeddings import OpenRouterEmbeddingClient
from agent_completion import agent_complete
import math

class RecursiveChunk:
    max_words: int = 40
    overlap: int = 5

    @classmethod
    def _resolve_config(cls, max_words=None, overlap=None):
        if max_words is None:
            max_words = cls.max_words

        if overlap is None:
            overlap = cls.overlap

        return max_words, overlap

    @staticmethod
    def _clean(
            parts: list[str]
            )-> list[str]:
        """Strip and remove empty strings from any list of text parts."""
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _fits(
            text: str, 
            max_words: int
            ) -> bool:
        """Check if text is within the word limit."""
        return len(text.split()) <= max_words

    @classmethod
    def chunk_by_paragraph(
            cls,
            text: str
            ) -> list[str]:
        return cls._clean(text.split('\n\n'))

    @classmethod
    def chunk_by_sentence(
            cls,
            text: str
            ) -> list[str]:
        return [s + "." for s in cls._clean(text.split("."))]

    @staticmethod
    def chunk_by_words(
            text: str, 
            chunk_size: int, 
            overlap: int
            ) -> list[str]:
        words = text.split()
        step = chunk_size - overlap
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]
    
    @classmethod
    def chunk(
        cls, 
        text, 
        max_words=None, 
        overlap=None):
        max_words, overlap = cls._resolve_config(max_words, overlap)

        final_chunks = []

        for paragraph in cls.chunk_by_paragraph(text):
            if cls._fits(paragraph, max_words):
                final_chunks.append(paragraph)
                continue

            for sentence in cls.chunk_by_sentence(paragraph):
                if cls._fits(sentence, max_words):
                    final_chunks.append(sentence)
                else:
                    final_chunks.extend(
                        cls.chunk_by_words(
                            sentence,
                            chunk_size=max_words,
                            overlap=overlap,
                        )
                    )

        return final_chunks


class RAGEngine:
    TOP_K: int = 2

    @classmethod
    def embed_chunks(
            cls,
            chunks: list[str]
            ):
        embeddings = OpenRouterEmbeddingClient.embed_texts(chunks)
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
        ) -> list[tuple[float, int, str]]:
        question_embedding = OpenRouterEmbeddingClient.embed_text(question)
        results = []

        for i, chunk in enumerate(chunks):
            score = cls.cosine_similarity(question_embedding, embeddings[i])
            results.append((score,i, chunk))

        results.sort(reverse=True)
        return results[:top_k]
    
    @classmethod
    def build_context(
        cls, 
        results: list[tuple[float, int, str]]
        ):
        context_parts = []
        for score, chunk_id, chunk in results:
            context_parts.append(f"[Chunk {chunk_id + 1}]\n{chunk}")

        context = "\n\n".join(context_parts)
        return context
    
    @classmethod
    def generate_answer(
            cls,
            question: str,
            context: str
            ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict RAG answerer. "
                    "Answer using only the provided context. "
                    "Mention the chunk label you used, like [Chunk 1]. "
                    "Do not use outside knowledge. "
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
        print(f"response: {answer}")
        return answer

    @classmethod
    def print_retrieval_debug(
        cls, 
        top_results : list[tuple[float, int, str]]
        ) -> None:
        print("Top results:")

        for score, chunk_id, chunk in top_results:
            print(f"Chunk {chunk_id + 1} score={score:.3f}")
            print(chunk)
            print()

def run_pipeline(text: str, question: str, top_k: int = 2):
    # 1. Chunk the document text
    chunks = RecursiveChunk.chunk(text)

    # 2. Generate embeddings for the chunks
    embeddings = RAGEngine.embed_chunks(chunks)

    # 3. Retrieve the top K matching chunks
    top_results = RAGEngine.retrieve(question, chunks, embeddings, top_k)


    # 4. Build context with fallback logic
    context = RAGEngine.build_context(top_results)
    answer = RAGEngine.generate_answer(question, context)
    # 5. Build prompt messages using the retrieved context
    RAGEngine.print_retrieval_debug(top_results)

    return answer


# --- To run the pipeline, you would call it like this: ---
if __name__ == "__main__":
    doc_text = """
                RAG helps language models answer questions using documents instead of only memory.

                Chunking splits a long document into smaller parts. Smaller chunks are easier to search, but they may lose context if they are too short.

                Overlap repeats some words from the previous chunk. This helps preserve meaning when an important idea crosses a chunk boundary.

                Similarity search compares the user's question with each chunk. The chunks with the highest scores are treated as the most relevant.

                Generation is the final step. The language model receives the retrieved chunks as context and writes an answer based on that context.
                """
    user_query = "Why do we use overlap when chunking?"
    
    run_pipeline(text=doc_text, question=user_query, top_k=2)
    print("PARAGRAPH CHUNKS")
    for chunk in RecursiveChunk.chunk_by_paragraph(doc_text):
        print("---")
        print(chunk)

    print("SENTENCE CHUNKS")
    for chunk in RecursiveChunk.chunk_by_sentence(doc_text):
        print("---")
        print(chunk)

    print("WORD CHUNKS")
    for chunk in RecursiveChunk.chunk_by_words(doc_text, chunk_size=25, overlap=5):
        print("---")
        print(chunk)

    print("RECURSIVE CHUNKS")
    for chunk in RecursiveChunk.chunk(doc_text, max_words=25, overlap=5):
        print("---")
        print(chunk)

