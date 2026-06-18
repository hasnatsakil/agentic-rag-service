
from vector_store import NeonVectorStore
from chat_service import ChatService


# ── Tuning parameters ─────────────────────────────────────
SEARCH_K = 15          # How many candidates to fetch from pgvector
ANSWER_K = 5          # Max chunks to pass to the LLM
MIN_SCORE = 0.3       # Cosine-similarity threshold (raise → stricter, lower → more lenient)
MAX_CONTEXT_CHARS = 5000  # Hard cap on total context length sent to LLM
# ──────────────────────────────────────────────────────────


def choose_document() -> int:
    documents = NeonVectorStore.list_documents()

    print("Documents:")

    for doc in documents:
        print(f"  {doc['id']}: {doc['file_name']} (created at {doc['created_at']})")

    document_id = int(input("\nChoose document id: "))

    return document_id

def main():
    document_id = choose_document()

    print()
    print("═" * 54)
    print("  RAG Chat — Level 5B Manual Test")
    print("═" * 54)
    print(f"  search_k        : {SEARCH_K}")
    print(f"  answer_k        : {ANSWER_K}")
    print(f"  min_score       : {MIN_SCORE}")
    print(f"  max_context_chars: {MAX_CONTEXT_CHARS}")
    print("═" * 54)
    print("  Type 'exit' or 'quit' to stop.")
    print()

    while True:
        question = input("You: ")
        
        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        result = ChatService.ask_pdf(
            question=question,
            document_id=document_id,
            search_k=SEARCH_K,
            answer_k=ANSWER_K,
            min_score=MIN_SCORE,
            max_context_chars=MAX_CONTEXT_CHARS,
        )

        print()
        print("Assistant:", result.answer)

        if result.sources:
            print("\nSources used:")
            for source in result.sources:
                print(f"  - {source.label()}, Score {source.score:.4f}")

        print()

if __name__ == "__main__":
    main()