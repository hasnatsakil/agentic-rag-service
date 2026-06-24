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
from core.models import RetrievalResult
from config import settings
from services.agent_completion import grading_complete

load_dotenv()

# State
class RAGState(TypedDict):
    question: str
    search_question: str
    DOCUMENT_ID: int
    SEARCH_K: int
    ANSWER_K: int
    MIN_SCORE: float
    MAX_CONTEXT_CHARS: int
    chunks : str
    answer: str
    filtered_results: list[RetrievalResult]
    selected_results: list[RetrievalResult]
    has_context: bool
    retry_count: int
    max_retries: int
    used_rewrite: bool
    is_grounded: bool


# Nodes
def retrieve_node(
        state: RAGState,
    ) -> dict:
    search_query = state["search_question"]
    query_embedding = OpenRouterEmbeddingClient.embed_query(search_query)

    results = NeonVectorStore.similarity_search(
        query_embedding=query_embedding,
        top_k=state["SEARCH_K"],
        DOCUMENT_ID=state["DOCUMENT_ID"],
    )
    filtered = [r for r in results if r.score >= state["MIN_SCORE"]]

    if not filtered:
        return {
          "filtered_results": [],
            "has_context": False
  
            }
    
    return {
        "filtered_results": filtered,
        "has_context": True
        }

def grade_documents_node(
        state: RAGState,
    ) -> dict:
    # If retrieve_node already filed, just pass the failure along
    if not state.get("has_context", False):
        return {"has_context": False}

    #temporarily build context just for the grader to evaluate
    temp_context = RAGEngine.build_context(state["filtered_results"])
    messages = [
        {
            "role" : "system",
            "content": "You are a grader assessing relevance of a retrieved document to a user question. "
            "If the document contains keyword(s) or semenatic meaning related to the user question, grade it as relevant. "
            "Give a binary score 'yes' or 'no' score to indicate wheather the document is relevant to the question."
        },
        {
            "role" : "user",
            "content": f"Retrived document: \n\n {temp_context} \n\n User question: {state['question']}"
        }
    ]

    completion = grading_complete(
        messages = messages,
        max_tokens=10
        )
    grade = completion.choices[0].message.content.strip().lower()

    if "yes" in grade:
        return {"has_context": True}
    else:
        return {"has_context": False}

def rewrite_query_node(
        state: RAGState,
    ) -> dict:
    original_query = state["search_question"]

    new_query = original_query + "explained in detail"

    return {
        "search_question": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "used_rewrite": True
    }

def select_context_node(
        state: RAGState,
    ) -> dict:
    filtered_results = state.get("filtered_results", [])
    selected = []
    current_chars = 0

    for result in filtered_results:
        nex_size = len(result.chunk_text)

        if current_chars + nex_size > state["MAX_CONTEXT_CHARS"]:
            break
        
        selected.append(result)
        current_chars += nex_size

        if len(selected) >= state["ANSWER_K"]:
            break
    
    if not selected:
        return {
            "chunks": "",
            "selected_results": [],
            "has_context": False
        }
    
    context_str = RAGEngine.build_context(selected)
    return {
        "chunks": context_str,
        "selected_results": selected,
        "has_context": True
    }

def answer_node(
        state: RAGState,
    ) -> dict:
    answer = RAGEngine.generate_answer(
        question=state["question"],
        context=state["chunks"]
    )
    return {"answer": answer}

def check_hallucination_node(
        state: RAGState
    ) -> dict:
    messages = [
        {
            "role" : "system",
            "content": "You are a grader assesing wheather an LLM generation is grounded "
            "in/ supported by a set of retrived facts.  "
            "give a binary score 'yes' or 'no'."
        },
        {
            "role" : "user",
            "content": f"Set of facts: \n\n {state['chunks']} \n\n LLM generation: {state['answer']}"
        }
    ]
    completion = grading_complete(
        messages = messages,
        max_tokens=10
    )
    grade = completion.choices[0].message.content.strip().lower()

    if "yes" in grade:
        return {"is_grounded": True}
    else:
        return {"is_grounded": False}

def no_context_node(state: RAGState) -> dict:
    return {
        "answer": "I could not find enough relevant context in this document to answer that. "
                  f"Try lowering MIN_SCORE below {state.get('MIN_SCORE', 0.3)} or asking a more specific question.",
        "has_context": False,
        "selected_results": []
    }


#Router
def right_after_grading(
        state: RAGState,
    ) -> str:
    if state.get("has_context"):
        return "select_context"
    elif state.get("retry_count", 0) < state.get("max_retries", 1):
        return "rewrite_query"
    else:
        return "no_context"

def route_after_select(
        state: RAGState,
    ) -> str:
    return "answer" if state.get("has_context") else "no_context"

def route_after_hallucination(
        state: RAGState,
    ) -> str:
    if state.get("is_grounded"):
        return END
    else:
        state["answer"] = ("WARNING: I generated an answer, but I could not verify "
        "it against the source text. \n\n") + state['answer']
        return END

# Build Graph
graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("grade_documents", grade_documents_node)
graph_builder.add_node("rewrite_query", rewrite_query_node)
graph_builder.add_node("select_context", select_context_node)
graph_builder.add_node("answer", answer_node)
graph_builder.add_node("check_hallucination", check_hallucination_node)
graph_builder.add_node("no_context", no_context_node)

# Add Edge Start -> Retrieve -> Check Context -> (Answer or End)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    right_after_grading,
    {
        "select_context": "select_context",
        "rewrite_query": "rewrite_query",
        "no_context": "no_context"
    }
)
graph_builder.add_edge("rewrite_query", "retrieve")
graph_builder.add_conditional_edges(
    "select_context",
    route_after_select,
    {
        "answer": "answer",
        "no_context": "no_context"
    }
)
graph_builder.add_edge("answer", "check_hallucination")
graph_builder.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination,
    {
        END: END,
    }
)
graph_builder.add_edge("no_context", END)

# Compile Graph
rag_graph = graph_builder.compile()

if __name__ == "__main__":
    question = input("Ask a question about your PDF: ")

    result = rag_graph.invoke({
        "question": question,
        "search_question": question,
        "DOCUMENT_ID": settings.DOCUMENT_ID,
        "SEARCH_K": settings.SEARCH_K,
        "ANSWER_K": settings.ANSWER_K,
        "MIN_SCORE": settings.MIN_SCORE,
        "MAX_CONTEXT_CHARS": settings.MAX_CONTEXT_CHARS,
        "chunks": "",
        "answer": "",
        "filtered_results": [],
        "selected_results": [],
        "has_context": False,
        "retry_count": 0,
        "max_retries": 1,
        "used_rewrite": False,
        "is_grounded": False
    })
    
    print("\n========== ANSWER ==========")
    if result["answer"]:
        print(result["answer"])
    else:
        print("No relevant information found.")

    print("\n[DEBUG] Used Rewrite:", result.get("used_rewrite"))
    print("[DEBUG] Is Grounded:", result.get("is_grounded"))
    print("============================")