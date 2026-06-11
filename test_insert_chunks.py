from importlib.resources import path

from anyio import Path

from chunking import RecursiveChunk
from rag_engine import RAGEngine
from vector_store import NeonVectorStore

path = Path("thesis_notes.txt")
text = """
RAG helps language models answer questions using documents instead of only memory.

Chunking splits a long document into smaller parts. Smaller chunks are easier to search, but they may lose context if they are too short.

Overlap repeats some words from the previous chunk. This helps preserve meaning when an important idea crosses a chunk boundary.

Similarity search compares the user's question with each chunk. The chunks with the highest scores are treated as the most relevant.

Generation is the final step. The language model receives the retrieved chunks as context and writes an answer based on that context.
"""

chunks = RecursiveChunk.chunk(text)
embeddings = RAGEngine.embed_chunks(chunks)

document_id = NeonVectorStore.create_document(file_name = path.name)

NeonVectorStore.insert_chunks(
    document_id = document_id,
    chunks = chunks,
    embeddings = embeddings
)
print(path.name)
print(f"Inserted document id: {document_id}")
print(f"Inserted Chunks: {len(chunks)}")