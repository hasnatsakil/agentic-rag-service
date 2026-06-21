import sys
import os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from core.embeddings import OpenRouterEmbeddingClient
from core.vector_store import NeonVectorStore
from core.rag_engine import RAGEngine

load_dotenv()

DOCUMENT_ID = 1
MIN_SCORE = 0.3

# State
class RAGState(TypedDict):
    question: str
    chunks : str
    answer: str
    has_context: bool

# Nodes
def retrieve_node(
        state: RAGState,
    ) -> dict:
    question = state["question"]
    query_embedding = OpenRouterEmbeddingClient.embed_query(question)

    results = NeonVectorStore.similarity_search(
        query_embedding=query_embedding,
        top_k=5,
        document_id=DOCUMENT_ID,
    )
    filtered = [r for r in results if r.score >= MIN_SCORE]

    if not filtered:
        return {"chunks": "", "has_context": False}
    
    context = RAGEngine.build_context(filtered)
    
    return {"chunks": context, "has_context": True}

def answer_node(
        state: RAGState,
    ) -> dict:
    answer = RAGEngine.generate_answer(
        question=state["question"],
        context=state["chunks"]
    )
    return {"answer": answer}

#Router
def check_context(
        state: RAGState,
    ) -> str:
    return "answer" if state["has_context"] else END

# Build Graph
graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("answer", answer_node)

# Add Edge Start -> Retrieve -> Check Context -> (Answer or End)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve",
    check_context,
    {
        "answer": "answer",
        END: END
    }
)
graph_builder.add_edge("answer", END)

# Compile Graph
rag_graph = graph_builder.compile()

if __name__ == "__main__":
    question = input("Ask a question about your PDF: ")

    result = rag_graph.invoke({
        "question": question,
        "chunks": "",
        "answer": "",
        "has_context": False
    })
    
    print("\n========== ANSWER ==========")
    if result["answer"]:
        print(result["answer"])
    else:
        print("No relevant information found.")

    print("============================")