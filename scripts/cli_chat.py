"""
scripts/cli_chat.py
Interactive Terminal Chat Agent for the Agentic RAG Service.

Provides a full-featured command-line interface to chat with ingested PDF
documents directly from the terminal, using the complete LangGraph RAG
pipeline with stateful multi-turn memory.

The agent supports two execution modes:

- **Direct Mode** (default): Calls :class:`~services.graph_services.GraphService`
  and :class:`~core.chat_history_store.ChatHistoryStore` directly in Python.
  No HTTP server required — simply run the script.

- **API Mode** (``--api``): Sends requests to a running FastAPI server
  (local or deployed on Render).

Startup Flow:
    1. Silently load the indexed document catalog from the Neon database
       (used internally by the graph for document routing — not displayed).
    2. List existing chat sessions for selection (resume) or create a new one.
    3. Enter the interactive prompt loop.
       Type ``/docs`` at any time to display the document catalog.

In-Chat Commands:
    /sessions  — List and switch to any active chat session.
    /history   — View the full message log of the active session.
    /docs      — Display all indexed documents in the database.
    /rerank    — Toggle Pass 2 LLM-as-a-Judge re-ranking ON/OFF.
    /new       — Create a new session thread and switch to it.
    /help      — Display available commands.
    exit/quit  — Exit the agent cleanly.

Usage::

    # Direct mode (no server required)
    python scripts/cli_chat.py

    # API mode against a local uvicorn server
    python scripts/cli_chat.py --api

    # API mode against a live Render deployment
    python scripts/cli_chat.py --api --url https://agentic-rag-service.onrender.com
"""

import sys
import os
import time
import argparse

# ---------------------------------------------------------------------------
# Resolve the project root so all imports work regardless of working directory.
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from config import settings
from core.vector_store import NeonVectorStore
from core.chat_history_store import ChatHistoryStore
from services.graph_services import GraphService
from services.compaction_service import save_and_compact_workflow

# ---------------------------------------------------------------------------
# Terminal colour codes — no external libraries required.
# ---------------------------------------------------------------------------

GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


# ===========================================================================
#  Display helpers
# ===========================================================================

def banner() -> None:
    """Print the startup ASCII banner with mode information."""
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}   🤖  Agentic RAG Service — Terminal Chat Agent{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")


def print_help() -> None:
    """Print the available in-chat command reference."""
    print(f"\n{BOLD}  Available Commands:{RESET}")
    print(f"  {CYAN}/sessions{RESET}  — List active sessions & switch/resume a thread")
    print(f"  {CYAN}/history{RESET}   — View full message log of the current session")
    print(f"  {CYAN}/docs{RESET}      — List all indexed documents in the database")
    print(f"  {CYAN}/rerank{RESET}    — Toggle Pass 2 LLM-as-a-Judge re-ranking ON/OFF")
    print(f"  {CYAN}/new{RESET}       — Start a fresh chat session thread")
    print(f"  {CYAN}/help{RESET}      — Show this command reference")
    print(f"  {CYAN}exit / quit{RESET}— Exit the agent\n")


def display_documents(docs: list) -> None:
    """Print a formatted catalog of all documents stored in the vector database.

    Args:
        docs: List of document dicts returned by
            :meth:`~core.vector_store.NeonVectorStore.list_documents`.
            Each dict should contain ``id``, ``file_name``, ``summary``,
            and ``created_at`` keys.
    """
    if not docs:
        print(
            f"\n{YELLOW}⚠  No documents found in the database. "
            f"Upload a PDF via the API or Swagger UI (/docs) first.{RESET}"
        )
        return

    print(f"\n{BOLD}📚  Indexed Documents ({len(docs)}):{RESET}")
    for doc in docs:
        doc_id  = doc.get("id")
        fname   = doc.get("file_name", "unknown")
        created = str(doc.get("created_at", ""))[:19]
        summary = doc.get("summary", "")
        summary_str = f"  {DIM}{summary}{RESET}" if summary else ""
        print(
            f"  [{BOLD}ID {doc_id}{RESET}] {CYAN}{fname}{RESET} "
            f"{DIM}({created}){RESET}{summary_str}"
        )


def display_history(session_id: str) -> None:
    """Fetch and print the full chronological message log of a session.

    Retrieves both the raw message turns and the current running summary
    from :class:`~core.chat_history_store.ChatHistoryStore` and formats
    them for terminal display.

    Args:
        session_id: The session identifier whose history to display.
    """
    messages = ChatHistoryStore.get_all_session_messages(session_id)
    summary  = ChatHistoryStore.get_summary(session_id)

    print(f"\n{BOLD}📜  History — Session: {GREEN}{session_id}{RESET}")

    if summary:
        print(f"  {DIM}Running summary: {summary}{RESET}")

    print(f"  {DIM}{'─'*56}{RESET}")

    if not messages:
        print(f"  {DIM}(No messages yet in this session){RESET}")
        return

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            print(f"\n  {BOLD}{GREEN}You   > {RESET}{content}")
        else:
            print(f"  {BOLD}{CYAN}Agent > {RESET}{content}")
    print()


# ===========================================================================
#  Session management
# ===========================================================================

def select_or_create_session() -> str:
    """Interactively select an existing chat session or create a new one.

    Queries :meth:`~core.chat_history_store.ChatHistoryStore.list_sessions`
    to retrieve all active sessions and presents them in a numbered menu.
    The user can:

    - Type a number to **resume** an existing session (history and running
      summary will be loaded automatically on the next query).
    - Type ``N`` or leave blank to start a **new session** (with an
      auto-generated or custom ID).

    Returns:
        The selected or newly created session ID string.
    """
    try:
        sessions = ChatHistoryStore.list_sessions()
    except Exception as exc:
        print(f"{YELLOW}⚠  Could not load existing sessions: {exc}{RESET}")
        sessions = []

    print(f"\n{BOLD}💬  Chat Session Manager:{RESET}")

    if sessions:
        print(f"  {len(sessions)} active session thread(s) found:\n")
        for idx, sess in enumerate(sessions, start=1):
            sid        = sess.get("session_id", "?")
            last_active = str(sess.get("last_active_at", ""))[:19]   # key from DB
            print(
                f"  {BOLD}[{idx}]{RESET} {GREEN}{sid}{RESET}  "
                f"{DIM}last activity: {last_active}{RESET}"
            )
        print(f"\n  {BOLD}[N]{RESET} Start a new session thread")

        choice = input(
            f"\n{BOLD}  Select [1-{len(sessions)}] to resume, or N for new: {RESET}"
        ).strip().upper()

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(sessions):
                chosen = sessions[val - 1]["session_id"]
                print(f"  {GREEN}✔  Resuming session '{chosen}'{RESET}")
                return chosen

    # Fall through → create new session.
    custom = input(
        f"\n{BOLD}  Enter new session ID (press Enter to auto-generate): {RESET}"
    ).strip()

    session_id = custom if custom else f"cli-session-{int(time.time())}"
    print(f"  {GREEN}✔  New session started: '{session_id}'{RESET}")
    return session_id


# ===========================================================================
#  Query execution — Direct mode
# ===========================================================================

def query_direct(
    question: str,
    session_id: str,
    use_rerank: bool,
    docs: list,
) -> tuple:
    """Execute a RAG query directly through the service and graph layers.

    Retrieves chat history and running summary from the database, invokes
    the compiled LangGraph pipeline via :class:`~services.graph_services.GraphService`,
    then persists the new turn and updates the running summary by calling
    :func:`~services.compaction_service.save_and_compact_workflow`.

    Args:
        question:   The user's natural-language question.
        session_id: Active session identifier for history and summary lookup.
        use_rerank: Whether to activate the optional Pass 2 LLM re-ranking node.
        docs:       Pre-fetched document catalog forwarded to ``available_documents``
                    in the RAGState so the agent can select the correct document ID.

    Returns:
        A tuple of ``(result, elapsed_ms)`` where ``result`` is a
        :class:`~core.models.ChatResult` and ``elapsed_ms`` is the
        wall-clock latency in milliseconds.
    """
    history = ChatHistoryStore.get_last_20_message(session_id)
    summary = ChatHistoryStore.get_summary(session_id)

    start_t = time.time()
    result = GraphService.ask_pdf_with_graph(
        question=question,
        SEARCH_K=settings.SEARCH_K,
        GRADE_K=settings.GRADE_K,
        ANSWER_K=settings.ANSWER_K,
        MIN_SCORE=settings.MIN_SCORE,
        MAX_CONTEXT_CHARS=settings.MAX_CONTEXT_CHARS,
        use_llm_rerank=use_rerank,
        available_documents=docs,
        history=history,
        summary=summary,
    )
    elapsed_ms = round((time.time() - start_t) * 1000, 1)

    # Persist the turn and conditionally update running summary.
    save_and_compact_workflow(
        session_id=session_id,
        user_question=question,
        assistant_answer=result.answer,
        old_summary=summary,
    )

    return result, elapsed_ms


# ===========================================================================
#  Query execution — API mode
# ===========================================================================

def query_api(
    question: str,
    session_id: str,
    use_rerank: bool,
    api_url: str,
) -> tuple:
    """Execute a RAG query via HTTP POST to the running FastAPI server.

    Constructs a :class:`~schemas.QueryRequest`-compatible payload and
    posts it to ``POST /chat/query``.  The server handles history loading,
    graph execution, and background compaction internally.

    Args:
        question:   The user's natural-language question.
        session_id: Active session identifier forwarded in the request body.
        use_rerank: Whether to request Pass 2 LLM re-ranking in the payload.
        api_url:    Base URL of the running FastAPI server
                    (e.g. ``http://127.0.0.1:8000`` or a Render URL).

    Returns:
        A tuple of ``(result, elapsed_ms)`` where ``result`` is a duck-typed
        object with the same attributes as :class:`~core.models.ChatResult`,
        and ``elapsed_ms`` is the server-reported processing time.

    Raises:
        requests.HTTPError: If the server returns a non-2xx response.
        requests.ConnectionError: If the server is unreachable.
    """
    try:
        import requests
    except ImportError:
        print(f"{RED}✖  'requests' package not found. Run: pip install requests{RESET}")
        sys.exit(1)

    payload = {
        "session_id":      session_id,
        "question":        question,
        "SEARCH_K":        settings.SEARCH_K,
        "GRADE_K":         settings.GRADE_K,
        "ANSWER_K":        settings.ANSWER_K,
        "MIN_SCORE":       settings.MIN_SCORE,
        "MAX_CONTEXT_CHARS": settings.MAX_CONTEXT_CHARS,
        "use_llm_rerank":  use_rerank,
    }
    response = requests.post(
        f"{api_url}/chat/query",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    # Wrap the API response JSON into an object with the same attribute names
    # used by direct mode so the rendering code is identical for both modes.
    class _APIResult:
        """Thin wrapper that mirrors the ChatResult attribute interface."""
        def __init__(self, d: dict):
            self.answer          = d.get("answer", "")
            self.sources         = d.get("sources", [])
            self.used_rewrite    = d.get("debug", {}).get("used_rewrite", False)
            self.is_grounded     = d.get("debug", {}).get("is_grounded", True)
            self.retrieval_count = d.get("debug", {}).get("retrieval_count", 0)
            self.selected_count  = d.get("debug", {}).get("selected_count", 0)

    return _APIResult(data), data.get("process_time_ms", 0.0)


# ===========================================================================
#  Result rendering
# ===========================================================================

def render_result(result, elapsed_ms: float, api_mode: bool) -> None:
    """Format and print a ChatResult (direct) or API result to the terminal.

    Displays the generated answer, source chunk citations with scores,
    and execution debug metrics.

    Args:
        result:     A :class:`~core.models.ChatResult` or an ``_APIResult``
                    object returned by the query functions.
        elapsed_ms: Wall-clock latency in milliseconds measured by the caller.
        api_mode:   If ``True``, ``elapsed_ms`` is server-reported time from
                    the API response.
    """
    # Answer
    print(f"\n  {BOLD}{CYAN}Agent > {RESET}{result.answer}\n")

    # Source citations
    sources = result.sources
    if sources:
        print(f"  {DIM}── Sources Cited {'─'*40}{RESET}")
        for src in sources:
            if isinstance(src, dict):
                # API mode — sources arrive as plain dicts.
                label = src.get("label", "")
                score = src.get("score", 0.0)
                text  = src.get("chunk_text", "")[:120]
            else:
                # Direct mode — sources are RetrievalResult dataclasses.
                label = src.label()
                score = src.score
                text  = src.chunk_text[:120]
            print(
                f"  {CYAN}• {label}{RESET} "
                f"{DIM}(score: {score:.3f}){RESET} — \"{text}...\""
            )

    # Debug metrics
    grounded_str = (
        f"{GREEN}Yes{RESET}"
        if getattr(result, "is_grounded", True)
        else f"{YELLOW}⚠ Unverified{RESET}"
    )
    rewrite_str = (
        f"{YELLOW}Yes — query was expanded{RESET}"
        if getattr(result, "used_rewrite", False)
        else "No"
    )
    latency_label = "server" if api_mode else "wall-clock"
    print(
        f"  {DIM}── {elapsed_ms} ms ({latency_label}) | "
        f"Grounded: {grounded_str} | "
        f"Rewrite: {rewrite_str} ──{RESET}\n"
    )


# ===========================================================================
#  Main interactive loop
# ===========================================================================

def main() -> None:
    """Entry point — parse arguments, run startup flow, and enter chat loop.

    Parses ``--api`` and ``--url`` flags, displays the startup banner,
    silently loads the document catalog (required internally by the graph for
    document routing but not printed to the terminal), prompts for session
    selection, then blocks on an interactive ``input()`` prompt loop until the
    user types ``exit``/``quit`` or presses ``Ctrl+C``.

    The document catalog is intentionally not displayed at startup to keep the
    terminal clean.  Users can type ``/docs`` at any time to view it.
    """
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat agent for the Agentic RAG Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cli_chat.py\n"
            "  python scripts/cli_chat.py --api\n"
            "  python scripts/cli_chat.py --api --url https://agentic-rag-service.onrender.com\n"
        ),
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use FastAPI REST server instead of direct Python service calls.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server (only used with --api).",
    )
    args = parser.parse_args()

    api_mode = args.api
    api_url  = args.url.rstrip("/")

    # ── Startup ──────────────────────────────────────────────────────────
    banner()

    if api_mode:
        print(f"  {MAGENTA}Mode     : API Client → {api_url}{RESET}")
    else:
        print(f"  {MAGENTA}Mode     : Direct Pipeline (no HTTP server required){RESET}")

    print(f"  {DIM}Tip: Type /help to see all available commands.{RESET}")

    # Load document catalog silently — used in direct mode to pass
    # available_documents to the graph.  Not displayed at startup;
    # users can type /docs in the chat to view the catalog.
    docs = []
    try:
        store = NeonVectorStore()
        docs  = store.list_documents()
        doc_count = len(docs)
        print(f"  {DIM}Database ready — {doc_count} document(s) indexed. Type /docs to view.{RESET}")
    except Exception as exc:
        print(f"{RED}✖  Could not connect to database: {exc}{RESET}")
        print(f"{YELLOW}   Check your DATABASE_URL in .env and try again.{RESET}")
        sys.exit(1)

    # Session selector — resume or create new.
    session_id = select_or_create_session()
    use_rerank = False

    print_help()

    # ── Interactive prompt loop ───────────────────────────────────────────
    while True:
        try:
            prompt = (
                f"\n{BOLD}{GREEN}[{session_id}]{RESET} "
                f"{BOLD}You > {RESET}"
            )
            user_input = input(prompt).strip()

            if not user_input:
                continue

            cmd_lower = user_input.lower()

            # ── Commands ─────────────────────────────────────────────────
            if cmd_lower in ("exit", "quit"):
                print(f"\n{CYAN}  Goodbye! Exiting agent. 👋{RESET}\n")
                break

            elif cmd_lower == "/help":
                print_help()

            elif cmd_lower == "/docs":
                try:
                    docs = NeonVectorStore().list_documents()
                except Exception as exc:
                    print(f"{RED}✖  {exc}{RESET}")
                display_documents(docs)

            elif cmd_lower == "/sessions":
                session_id = select_or_create_session()

            elif cmd_lower == "/history":
                display_history(session_id)

            elif cmd_lower == "/new":
                new_id = f"cli-session-{int(time.time())}"
                session_id = new_id
                print(
                    f"  {GREEN}✔  Switched to new session '{session_id}'{RESET}\n"
                    f"  {DIM}Note: this session is saved to the database after your first message.{RESET}"
                )

            elif cmd_lower == "/rerank":
                use_rerank = not use_rerank
                state = f"{GREEN}ON{RESET}" if use_rerank else f"{RED}OFF{RESET}"
                print(f"  Pass 2 LLM Re-ranking → {state}")

            else:
                # ── RAG Query ─────────────────────────────────────────────
                print(f"  {DIM}Searching knowledge base & generating answer...{RESET}")

                if api_mode:
                    result, elapsed_ms = query_api(
                        user_input, session_id, use_rerank, api_url
                    )
                else:
                    result, elapsed_ms = query_direct(
                        user_input, session_id, use_rerank, docs
                    )

                render_result(result, elapsed_ms, api_mode)

        except KeyboardInterrupt:
            print(f"\n\n{CYAN}  Session interrupted. Exiting agent. 👋{RESET}\n")
            break
        except Exception as exc:
            print(f"\n  {RED}✖  Error: {exc}{RESET}")


if __name__ == "__main__":
    main()
