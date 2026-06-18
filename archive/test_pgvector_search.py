
from embeddings import OpenRouterEmbeddingClient
from vector_store import NeonVectorStore

question = "Why do we use overlap when chunking?"

query_embedding = OpenRouterEmbeddingClient.embed_query(question)

results = NeonVectorStore.similarity_search(
    query_embedding = query_embedding,
    top_k = 2
)
print("Top pgvector results:")

for score, chunk_index, chunk_text in results:
    print(f"Chunk {chunk_index + 1} score = {score:.3f}")
    print(chunk_text)
    print()