"""
LangGraph RAG pipeline definition.

This module builds and compiles the stateful LangGraph workflow that powers
the self-correcting, agentic Retrieval-Augmented Generation loop.  The graph
is compiled once at import time as :data:`rag_graph` and shared across all
requests.

Pipeline overview
-----------------
The graph implements the following flow::

    START
      │
      ▼
    [agent]  ─── direct answer ──► [check_hallucination] ──► END
      │ tool call requested
      ▼
    [execute_tool]
      │
      ├── use_llm_rerank=True ──► [llm_rerank] ──► [grade_documents]
      └── use_llm_rerank=False ─────────────────► [grade_documents]
                                                          │
                                  ┌───── relevant ────────┤
                                  │                       │
                          [select_context]        retry? ─┤
                                  │                       │
                                  └──► [agent]   [rewrite_query]
                                       (loop)          │
                                               [execute_tool] (retry)

    grade_documents exhausted: [no_context] ──► END

Node descriptions:
    agent_node:               Calls the LLM with tool schema + chat history +
                              running summary + document catalog.  Either
                              produces a direct answer or requests a vector
                              database search via native tool calling.
    execute_tool_node:        Embeds the search query, runs hybrid
                              (vector + keyword) search, fuses results via RRF,
                              and applies Pass 1 lexical keyword rescoring.
    llm_rerank_node:          Optional Pass 2 re-ranking. Sends candidate
                              chunks to a judge LLM which returns them in
                              best-first order before grading.
    grade_documents_node:     Sends each candidate chunk to a lightweight judge
                              LLM for binary relevance grading (yes / no).
    rewrite_query_node:       If no relevant chunks were found, an LLM rewrites
                              the query into keyword-style search terms before
                              retrying retrieval.
    select_context_node:      Trims the graded chunks to fit within the
                              ``MAX_CONTEXT_CHARS`` and ``ANSWER_K`` budget,
                              then routes back to agent_node.
    check_hallucination_node: Verifies the generated answer is supported by the
                              retrieved facts. Prepends a warning if not.
    no_context_node:          Returns a helpful message when no relevant context
                              was found after all retries are exhausted.

Can be run directly for interactive testing::

    python graph/rag_graph.py
"""

import sys
import os

# Ensure the project root is on sys.path when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from core.embeddings import OpenRouterEmbeddingClient
from core.vector_store import NeonVectorStore
from core.rag_engine import RAGEngine
from core.models import RetrievalResult
from config import settings
from services.agent_completion import agent_complete

load_dotenv()


# ------------------------------------------------------------------ #
#  Graph state                                                        #
# ------------------------------------------------------------------ #

class RAGState(TypedDict):
    """Typed state dictionary shared across all graph nodes.

    Each field is read and/or written by one or more nodes.  LangGraph
    merges node return dicts into this state between steps.

    Attributes:
        question (str): The original user question (immutable throughout
            the run).
        search_question (str): The current search query; may be rewritten
            by :func:`rewrite_query_node`.
        DOCUMENT_ID (int): ID of the target document in the vector store.
        SEARCH_K (int): Number of candidate chunks to retrieve per search.
        ANSWER_K (int): Maximum chunks forwarded to the LLM after selection.
        MIN_SCORE (float): Minimum retrieval score; lower-scoring chunks are
            filtered out after retrieval.
        MAX_CONTEXT_CHARS (int): Character budget for LLM context.
        chunks (str): The current context string built from selected results.
        answer (str): The LLM-generated answer (populated by
            :func:`answer_node` or :func:`agent_node`).
        filtered_results (list[RetrievalResult]): All chunks that passed the
            ``MIN_SCORE`` threshold, before size-based selection.
        selected_results (list[RetrievalResult]): The final subset of chunks
            forwarded to the LLM after :func:`select_context_node`.
        has_context (bool): ``True`` if relevant context was found and
            confirmed by grading.
        retry_count (int): Number of query-rewrite attempts so far.
        max_retries (int): Maximum allowed rewrites before falling back to
            :func:`no_context_node`.
        used_rewrite (bool): Whether a query rewrite occurred during this run.
        is_grounded (bool): Whether the generated answer was confirmed
            grounded in the retrieved facts by :func:`check_hallucination_node`.
    """

    question: str
    search_question: str
    DOCUMENT_ID: int
    available_documents: list[dict]
    history: list[dict]
    summary: str
    SEARCH_K: int
    GRADE_K:int
    ANSWER_K: int
    MIN_SCORE: float
    MAX_CONTEXT_CHARS: int
    chunks: str
    answer: str
    filtered_results: list[RetrievalResult]
    selected_results: list[RetrievalResult]
    has_context: bool
    retry_count: int
    max_retries: int
    used_rewrite: bool
    is_grounded: bool
    use_llm_rerank:bool


# ------------------------------------------------------------------ #
#  Nodes                                                              #
# ------------------------------------------------------------------ #

def agent_node(state: RAGState) -> dict:
    """Call the LLM and decide whether to answer directly or invoke a tool.

    Constructs a system + user message pair.  If context is already
    available in the state, it is appended as a second system message so
    the model can use it to answer directly.  The model is given the
    ``search_pdf_database`` tool schema; if it emits a tool call, the
    requested query is captured and returned so the graph routes to
    :func:`execute_tool_node`.

    Args:
        state: Current graph state.

    Returns:
        Either ``{"search_question": <query>, "used_rewrite": True}`` when
        the LLM requests a tool call, or ``{"answer": <text>}`` when the
        LLM answers directly.
    """
    doc_catalog = []
    for doc in state.get("available_documents", []):
        doc_catalog.append(
            f"ID: {doc['id']} | Filename: {doc['file_name']} | Scope: {doc.get('summary', '')}"
        )
    catalog_str = "\n".join(doc_catalog)

    history_str = ""
    for msg in state.get("history", []):
        history_str += f"{msg['role']}: {msg['content']}\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful document assistant. "
                "If the user is asking a question about a document, you must choose the most relevant "
                "document from the catalog and query the database using the search tool.\n"
                # ─── ADD THIS WARNING LINE ────────────────────────
                "You MUST use the exact integer 'ID' value (e.g., 10, 9, 8) as the 'document_id' argument. "
                "Do NOT use row numbers, sequence indices, or make up your own IDs.\n"
                "If the question is a greeting or a general follow-up that you can answer from "
                "your context or summary, answer directly. Do not use the tool "
                "if the answer is already on your context.\n\n"
                f"Available Documents:\n{catalog_str}\n\n"
                f"Running Memory of Older Chats: {state.get('summary', '')}\n\n"
                f"Recent cat history:\n{history_str}"
            ),
        },
        {
            "role": "user",
            "content": f"User question: {state['question']}",
        },
    ]

    # Inject already-retrieved context so the model can answer without another search.
    if state.get("has_context") and state.get("chunks"):
        messages.append({
            "role": "system",
            "content": f"Database search results: {state['chunks']}",
        })
    has_retrieved_context = bool(state.get("has_context") and state.get("chunks"))

    tool_to_pass = None if has_retrieved_context else RAGEngine.get_tools()

    completion = agent_complete(
        messages=messages,
        tools=tool_to_pass,
    )
    message = completion.choices[0].message

    # If the LLM decided to call a tool, extract the search query.
    if message.tool_calls:
        args = json.loads(message.tool_calls[0].function.arguments)

        selected_doc_id = args.get("document_id", state.get("DOCUMENT_ID"))
        return {
            "search_question": args.get("query", state["question"]),
            "DOCUMENT_ID": selected_doc_id,
            "used_rewrite": True,
        }
    else:
        return {"answer": message.content}


def execute_tool_node(state: RAGState) -> dict:
    """Execute a hybrid vector + keyword search and build the context string.

    Embeds ``search_question``, runs
    :meth:`~core.vector_store.NeonVectorStore.hybrid_search`, filters
    results below ``MIN_SCORE``, and assembles the context string via
    :meth:`~core.rag_engine.RAGEngine.build_context`.

    Args:
        state: Current graph state.

    Returns:
        A dict updating ``filtered_results``, ``has_context``, and ``chunks``.
        If no results pass the score threshold, ``has_context`` is ``False``
        and ``chunks`` contains a "not found" message.
    """
    print(f"[TOOL EXECUTION] Searching database for: {state['search_question']}")

    query_embedding = OpenRouterEmbeddingClient.embed_query(state["search_question"])

    results = NeonVectorStore.hybrid_search(
        query=state["search_question"],
        query_embedding=query_embedding,
        top_k=state["SEARCH_K"],
        vector_k=state["SEARCH_K"],
        keyword_k=state["SEARCH_K"],
        DOCUMENT_ID=state["DOCUMENT_ID"],
    )

    results = RAGEngine.rerank_results(
        query=state["search_question"],
        results=results,
    )

    print("[GRAPH RETRIEVAL] total results:", len(results))

    for result in results:
        print(
            "[GRAPH RETRIEVAL]",
            result.label(),
            result.retrieval_method,
            result.score,
            result.chunk_text[:120].replace("\n", " ")
        )

    # Filter out chunks below the minimum score threshold.
    filtered = [r for r in results if r.score >= state["MIN_SCORE"]]
    print("[GRAPH RETRIEVAL] filtered results:", len(filtered))

    if not filtered:
        return {
            "filtered_results": [],
            "has_context": False,
            "chunks": "No relevant information found in the database.",
        }

    context_str = RAGEngine.build_context(filtered)
    return {
        "filtered_results": filtered,
        "has_context": True,
        "chunks": context_str,
    }

def llm_rerank_node(state: RAGState) -> dict:
    """Re-rank candidate chunks using an LLM-as-a-Judge (Pass 2 Re-ranking).

    Passes retrieved candidate chunks to a fast model that evaluates overall semantic fit
    to the user question and returns a best-first ordered list of chunk indexes.
    Re-orders ``filtered_results`` accordingly.

    Args:
        state: Current graph state containing ``filtered_results``.

    Returns:
        A dict updating ``filtered_results`` with the LLM-re-ordered results.
    """
    results = state.get("filtered_results", [])

    if not results:
        return {"filtered_results": []}
    
    candidate_text = []

    for index, result in enumerate(results, start=1):

        candidate_text.append(
            f"{index}. {result.label()}\n{result.chunk_text[:800]}"
        )
    messages = [
        {
            "role":"system",
            "content": (
                "You are a retrieval reranker. "
                "Given a user question and retrieved chunks, return only the numbers "
                "of the chunks that are most relevant, in best-first order. "
                "Do not explain."
            ),
        },
        {
            "role":"user",
            "content": (
                f"Question:\n{state['question']}\n\n"
                f"Chunks\n\n{chr(10).join(candidate_text)}\n\n"
                f"Return chunk numbers only, seperated by commas, Example: 3,1,5"
            ),
        },
    ]

    completion = agent_complete(
        messages=messages,
        max_tokens=500,
        is_grading=True
    )

    content = completion.choices[0].message.content or ""
    print("[LLM RERANK RAW]", content)

    chosen_indexes = []
    import re

    for match in re.findall(r"\d+", content):
        chosen_indexes.append(int(match))

    if not chosen_indexes:
        print("[LLM RERANK] No valid indexes returned; keeping original order.")
        return {"filtered_results": results}

    seen = set()
    dedup_indexes = []

    for index in chosen_indexes:
        if index not in seen:
            seen.add(index)
            dedup_indexes.append(index)
    chosen_indexes = dedup_indexes
    
    reranked = []

    for chosen_index in chosen_indexes:
        list_index = chosen_index - 1

        if 0 <= list_index < len(results):
            reranked.append(results[list_index])
        
    seen_ids = {(r.page_number, r.chunk_id) for r in reranked}

    for result in results:
        key = (result.page_number, result.chunk_id)
        if key not in seen_ids:
            reranked.append(result)
    
    for result in reranked:
        result.retrieval_method = f"{result.retrieval_method}+llm_rerank"

    return {"filtered_results": reranked}


def grade_documents_node(state: RAGState) -> dict:
    """Grade the retrieved context for relevance using a lightweight LLM.

    Skips grading immediately if no context was found.  Otherwise builds a
    temporary context string and asks the grading model for a binary
    ``"yes"`` / ``"no"`` relevance verdict.

    Args:
        state: Current graph state.

    Returns:
        ``{"has_context": True}`` if the context is judged relevant,
        ``{"has_context": False}`` otherwise (triggering a retry or fallback).
    """
    # Short-circuit: nothing to grade if retrieval already failed.
    if not state.get("has_context", False):
        return {
            "has_context": False,
            "filtered_results": []
            }
    candidates_to_grade = state["filtered_results"][: state["GRADE_K"]]

    relavant_results = []
    for result in candidates_to_grade:
        messages = [
            {
                "role":"system",
                "content": (
                    "You are a strict relevance grader. "
                    "Decide whether this retrieved chunk is relevant to the user's question.  "
                    "Answer only 'yes' or 'no'."
                ),
            },
            {
                "role":"user",
                "content": (
                    f"Question:\n{state['question']}\n\n"
                    f"chunks:\n{result.chunk_text}"
                ),
            },
        ]

        completion = agent_complete(
            messages=messages,
            max_tokens=300,
            temperature=0.0,
            is_grading=True,
        )
        message = completion.choices[0].message
        content = message.content or getattr(message, "reasoning", "")
        # content = completion.choices[0].message.content
        grade = content.strip().lower() if content else "no"

        print("[CHUNK GRADER]", result.label(), grade)

        if "yes" in grade:
            relavant_results.append(result)
    
    if not relavant_results:
        return {
            "has_context": False,
            "filtered_results": [],
        }
    return {
        "has_context":True,
        "filtered_results": relavant_results,
    }


def rewrite_query_node(state: RAGState) -> dict:
    """Expand the search query before a retry.

    Appends ``"explained in detail"`` to the current search question as a
    simple expansion heuristic.  Increments :attr:`RAGState.retry_count`
    and sets :attr:`RAGState.used_rewrite` to ``True``.

    Args:
        state: Current graph state.

    Returns:
        A dict updating ``search_question``, ``retry_count``, and
        ``used_rewrite``.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's question into a short keyword-style search query "
                "for finding relevant PDF chunks. Keep important exact terms. "
                "Do not answer the question. Return only the rewritten query."
            ),
        },
        {
            "role": "user",
            "content": state["question"],
        },
    ]

    completion = agent_complete(
        messages=messages,
        max_tokens=100,
        temperature=0.0,
        is_grading=True,
    )

    new_query = completion.choices[0].message.content or state["search_question"]
    new_query = new_query.strip()
    print("[QUERY REWRITE]", state["search_question"], "->", new_query)

    return {
        "search_question": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "used_rewrite": True,
        "filtered_results": [],
        "selected_results": [],
        "chunks": "",
        "has_context": False,
    }



def select_context_node(state: RAGState) -> dict:
    """Trim retrieved chunks to fit within budget constraints.

    Iterates through ``filtered_results`` in score order and appends chunks
    to the selection until either the ``MAX_CONTEXT_CHARS`` character budget
    or the ``ANSWER_K`` chunk limit is reached.

    Args:
        state: Current graph state.

    Returns:
        A dict updating ``chunks``, ``selected_results``, and ``has_context``.
        When no chunks fit the budget, ``has_context`` is set to ``False``.
    """
    filtered_results = state.get("filtered_results", [])
    selected: list[RetrievalResult] = []
    current_chars = 0

    for result in filtered_results:
        next_size = len(result.chunk_text)

        # Stop if adding this chunk would exceed the character budget.
        if current_chars + next_size > state["MAX_CONTEXT_CHARS"]:
            break

        selected.append(result)
        current_chars += next_size

        # Stop once the chunk count limit is reached.
        if len(selected) >= state["ANSWER_K"]:
            break

    if not selected:
        return {
            "chunks": "",
            "selected_results": [],
            "has_context": False,
        }

    context_str = RAGEngine.build_context(selected)
    return {
        "chunks": context_str,
        "selected_results": selected,
        "has_context": True,
    }


def check_hallucination_node(state: RAGState) -> dict:
    """Verify that the generated answer is grounded in the retrieved facts.

    Uses a dedicated hallucination-detection model to produce a binary
    ``"yes"`` / ``"no"`` verdict.  When the answer is not grounded, the
    route function :func:`route_after_hallucination` prepends a warning to
    the answer rather than blocking the response entirely.

    Args:
        state: Current graph state (both ``chunks`` and ``answer`` must be set).

    Returns:
        ``{"is_grounded": True}`` or ``{"is_grounded": False}``.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a grader assessing whether an LLM generation is grounded in / "
                "supported by a set of retrieved facts. "
                "Give a binary score 'yes' or 'no'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Set of facts:\n\n {state['chunks']} \n\n"
                f"LLM generation: {state['answer']}"
            ),
        },
    ]

    completion = agent_complete(
        messages=messages,
        max_tokens=200,
        temperature=0.0,
        is_hallucination=True,
    )
    content = completion.choices[0].message.content
    grade = content.strip().lower() if content else "no"

    if "yes" in grade:
        return {"is_grounded": True}
    else:
        return {"is_grounded": False}


def no_context_node(state: RAGState) -> dict:
    """Return a user-friendly message when no relevant context was found.

    Called when retrieval + grading exhausts all retries without finding
    relevant content.  Guides the user toward adjusting their query or
    the ``MIN_SCORE`` threshold.

    Args:
        state: Current graph state.

    Returns:
        A dict setting ``answer``, ``has_context`` (``False``), and
        ``selected_results`` (empty list).
    """
    return {
        "answer": (
            "I could not find enough relevant context in this document to answer that. "
            f"Try asking a more specific question or increasing SEARCH_K."
            "or asking a more specific question."
        ),
        "has_context": False,
        "selected_results": [],
    }


# ------------------------------------------------------------------ #
#  Routing functions                                                 #
# ------------------------------------------------------------------ #

def right_after_grading(state: RAGState) -> str:
    """Route after the grading node based on relevance and retry budget.

    Args:
        state: Current graph state.

    Returns:
        ``"select_context"`` if the context is relevant,
        ``"rewrite_query"`` if retries remain,
        ``"no_context"`` if the retry budget is exhausted.
    """
    if state.get("has_context"):
        return "select_context"
    elif state.get("retry_count", 0) < state.get("max_retries", 1):
        return "rewrite_query"
    else:
        return "no_context"


def route_after_agent(state: RAGState) -> str:
    """Route after the agent node based on whether an answer was produced.

    Args:
        state: Current graph state.

    Returns:
        ``END`` if the agent produced a direct answer,
        ``"execute_tool"`` if it requested a database search.
    """
    if state.get("answer"):
        return "check_hallucination"
    else:
        return "execute_tool"

def route_after_tool(state:RAGState) -> str:
    if not state.get("has_context"):
        return "grade_documents"

    if state.get("use_llm_rerank"):
        return "llm_rerank"

    return "grade_documents"


def route_after_hallucination(state: RAGState) -> str:
    """Route after the hallucination check, optionally prepending a warning.

    When the answer is not grounded, a warning prefix is injected directly
    into the state's ``answer`` field before routing to ``END``, so the
    caller always receives a response.

    Args:
        state: Current graph state.

    Returns:
        Always ``END``.
    """
    if state.get("is_grounded"):
        return END
    else:
        # Prepend a warning but still return the answer — don't block the user.
        answer = state.get("answer") or ""
        state["answer"] = (
            "WARNING: I generated an answer, but I could not verify it "
            "against the source text.\n\n"
        ) + answer
        return END


# ------------------------------------------------------------------ #
#  Graph construction                                                 #
# ------------------------------------------------------------------ #

graph_builder = StateGraph(RAGState)

# Register all nodes.
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("execute_tool", execute_tool_node)
graph_builder.add_node("grade_documents", grade_documents_node)
graph_builder.add_node("rewrite_query", rewrite_query_node)
graph_builder.add_node("select_context", select_context_node)
graph_builder.add_node("check_hallucination", check_hallucination_node)
graph_builder.add_node("no_context", no_context_node)
graph_builder.add_node("llm_rerank", llm_rerank_node)

# Wire up edges:
# START → agent → (answer directly or execute tool)
graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "check_hallucination": "check_hallucination",
        "execute_tool": "execute_tool",
    }
)

# execute_tool → grade → (select context | rewrite | no context)
graph_builder.add_conditional_edges(
    "execute_tool",
    route_after_tool,
    {
        "llm_rerank": "llm_rerank",
        "grade_documents": "grade_documents",
    }
)

graph_builder.add_edge("llm_rerank", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    right_after_grading,
    {
        "select_context": "select_context",
        "rewrite_query": "rewrite_query",
        "no_context": "no_context",
    },
)

# Retry loop: rewrite → execute_tool.
graph_builder.add_edge("rewrite_query", "execute_tool")

# Successful path: select_context → agent (now has context, should answer).
graph_builder.add_edge("select_context", "agent")

# Hallucination check → END (always, warning injected in route fn if ungrounded).
graph_builder.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination,
    {END: END},
)

# No context → END.
graph_builder.add_edge("no_context", END)

#: Compiled, thread-safe graph instance shared across all requests.
rag_graph = graph_builder.compile()


# ------------------------------------------------------------------ #
#  Interactive CLI entry point                                        #
# ------------------------------------------------------------------ #

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
        "is_grounded": False,
        "use_llm_rerank": False,
    })

    print("\n========== ANSWER ==========")
    if result["answer"]:
        print(result["answer"])
    else:
        print("No relevant information found.")

    print("\n[DEBUG] Used Rewrite:", result.get("used_rewrite"))
    print("[DEBUG] Is Grounded:", result.get("is_grounded"))
    print("============================")
