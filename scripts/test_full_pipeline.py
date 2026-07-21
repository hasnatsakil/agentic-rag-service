#!/usr/bin/env python3
"""
scripts/test_full_pipeline.py
Full end-to-end pipeline test runner for the PDF RAG Chat API.

Tests every endpoint in the correct order with colored terminal output,
assertion checks, and automatic cleanup even on failure.

Prerequisites:
    1. Server must be running:  uvicorn api:app --reload
    2. A sample PDF must exist:  data/sample.pdf
    3. The 'requests' package must be installed: pip install requests

Usage:
    python scripts/test_full_pipeline.py
    python scripts/test_full_pipeline.py --base-url http://your-render-url.com
"""

import sys
import time
import argparse

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not found. Run: pip install requests")
    sys.exit(1)

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SAMPLE_PDF_PATH  = "data/sample.pdf"
TEST_SESSION_ID  = f"test-session-{int(time.time())}"
TEST_QUESTION    = "What is this document about?"

# ------------------------------------------------------------------ #
#  Terminal colours (no external dependencies)                        #
# ------------------------------------------------------------------ #

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔  {msg}{RESET}")
def fail(msg):  print(f"  {RED}✖  {msg}{RESET}")
def info(msg):  print(f"  {CYAN}→  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠  {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}");\
                print(f"{BOLD}{CYAN}  {msg}{RESET}");\
                print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

# ------------------------------------------------------------------ #
#  Test state                                                          #
# ------------------------------------------------------------------ #

passed  = 0
failed  = 0
doc_id  = None   # Filled after upload so cleanup can delete it.


def assert_status(response, expected: int, label: str):
    """Assert HTTP status code and increment pass/fail counters."""
    global passed, failed
    if response.status_code == expected:
        ok(f"{label} — HTTP {response.status_code}")
        passed += 1
        return True
    else:
        fail(f"{label} — Expected HTTP {expected}, got {response.status_code}")
        try:
            fail(f"    Body: {response.json()}")
        except Exception:
            fail(f"    Body: {response.text[:200]}")
        failed += 1
        return False


def assert_key(data: dict, key: str, label: str):
    """Assert a key exists in the response JSON."""
    global passed, failed
    if key in data:
        ok(f"{label} — key '{key}' present")
        passed += 1
        return True
    else:
        fail(f"{label} — key '{key}' missing from response")
        failed += 1
        return False


# ================================================================== #
#  Test steps                                                         #
# ================================================================== #

def test_health(base: str):
    header("TEST 1 — Health Check  GET /health")
    t0 = time.time()
    try:
        r = requests.get(f"{base}/health", timeout=10)
    except requests.ConnectionError:
        fail(f"Cannot connect to {base}. Is the server running?")
        sys.exit(1)

    ms = round((time.time() - t0) * 1000, 1)
    if assert_status(r, 200, "GET /health"):
        data = r.json()
        info(f"status   = {data.get('status')}")
        info(f"database = {data.get('database')}")
        info(f"latency  = {ms} ms")
        if data.get("status") != "ok":
            warn("Database is not healthy — subsequent tests may fail.")


def test_upload(base: str) -> int | None:
    header("TEST 2 — PDF Upload  POST /documents/upload")
    try:
        with open(SAMPLE_PDF_PATH, "rb") as f:
            t0 = time.time()
            r = requests.post(
                f"{base}/documents/upload",
                files={"file": (SAMPLE_PDF_PATH, f, "application/pdf")},
                timeout=60,
            )
            ms = round((time.time() - t0) * 1000, 1)
    except FileNotFoundError:
        fail(f"Sample PDF not found at '{SAMPLE_PDF_PATH}'. Cannot test upload.")
        return None

    if assert_status(r, 200, "POST /documents/upload"):
        data = r.json()
        assert_key(data, "DOCUMENT_ID", "Upload response")
        assert_key(data, "chunk_count", "Upload response")
        info(f"DOCUMENT_ID  = {data.get('DOCUMENT_ID')}")
        info(f"file_name    = {data.get('file_name')}")
        info(f"page_count   = {data.get('page_count')}")
        info(f"chunk_count  = {data.get('chunk_count')}")
        info(f"latency      = {ms} ms")
        return data.get("DOCUMENT_ID")
    return None


def test_list_documents(base: str):
    header("TEST 3 — List Documents  GET /documents")
    t0 = time.time()
    r = requests.get(f"{base}/documents", timeout=10)
    ms = round((time.time() - t0) * 1000, 1)

    if assert_status(r, 200, "GET /documents"):
        data = r.json()
        assert_key(data, "documents", "List documents response")
        count = len(data.get("documents", []))
        info(f"Total documents in DB = {count}")
        info(f"latency               = {ms} ms")
        if count == 0:
            warn("No documents found — chat query test may return 400.")


def test_chat_query(base: str):
    header("TEST 4 — Chat Query  POST /chat/query")
    payload = {
        "session_id":     TEST_SESSION_ID,
        "question":       TEST_QUESTION,
        "SEARCH_K":       15,
        "GRADE_K":        6,
        "ANSWER_K":       3,
        "MIN_SCORE":      0.0,
        "MAX_CONTEXT_CHARS": 3000,
        "use_llm_rerank": False,
    }
    info(f"session_id = {TEST_SESSION_ID}")
    info(f"question   = {TEST_QUESTION}")

    t0 = time.time()
    r = requests.post(f"{base}/chat/query", json=payload, timeout=120)
    ms = round((time.time() - t0) * 1000, 1)

    if assert_status(r, 200, "POST /chat/query"):
        data = r.json()
        assert_key(data, "answer",         "Query response")
        assert_key(data, "sources",        "Query response")
        assert_key(data, "process_time_ms","Query response")
        assert_key(data, "debug",          "Query response")

        answer = data.get("answer", "")
        debug  = data.get("debug", {})
        info(f"answer (first 120 chars) = {answer[:120]}...")
        info(f"sources returned         = {len(data.get('sources', []))}")
        info(f"retrieval_count          = {debug.get('retrieval_count')}")
        info(f"selected_count           = {debug.get('selected_count')}")
        info(f"is_grounded              = {debug.get('is_grounded')}")
        info(f"used_rewrite             = {debug.get('used_rewrite')}")
        info(f"process_time_ms          = {data.get('process_time_ms')} ms")
        info(f"total wall-clock latency = {ms} ms")

        if not answer.strip():
            warn("Answer is empty — possible retrieval or LLM issue.")


def test_multi_turn_query(base: str):
    """Send a second question in the same session to verify memory works."""
    header("TEST 5 — Multi-Turn Memory  POST /chat/query (turn 2)")
    payload = {
        "session_id":     TEST_SESSION_ID,
        "question":       "Can you summarise what you just told me?",
        "SEARCH_K":       15,
        "GRADE_K":        6,
        "ANSWER_K":       3,
        "MIN_SCORE":      0.0,
        "MAX_CONTEXT_CHARS": 3000,
        "use_llm_rerank": False,
    }
    info(f"session_id = {TEST_SESSION_ID}  (same session — should recall previous turn)")

    t0 = time.time()
    r = requests.post(f"{base}/chat/query", json=payload, timeout=120)
    ms = round((time.time() - t0) * 1000, 1)

    if assert_status(r, 200, "POST /chat/query (turn 2)"):
        data = r.json()
        info(f"answer (first 120 chars) = {data.get('answer','')[:120]}...")
        info(f"process_time_ms          = {data.get('process_time_ms')} ms")
        info(f"total wall-clock latency = {ms} ms")


def test_list_sessions(base: str):
    header("TEST 6 — List Sessions  GET /chat/sessions")
    # Small delay to let background compaction task complete.
    time.sleep(1)
    r = requests.get(f"{base}/chat/sessions", timeout=10)

    if assert_status(r, 200, "GET /chat/sessions"):
        data = r.json()
        assert_key(data, "sessions", "Sessions list response")
        sessions = data.get("sessions", [])
        info(f"Total active sessions = {len(sessions)}")
        found = any(
            s.get("session_id") == TEST_SESSION_ID
            for s in sessions
        )
        if found:
            ok(f"Test session '{TEST_SESSION_ID}' is visible in session list")
        else:
            warn(f"Test session '{TEST_SESSION_ID}' not found (may still be persisting via background task)")


def test_get_session_messages(base: str):
    header("TEST 7 — Session Messages  GET /chat/sessions/{session_id}/messages")
    r = requests.get(
        f"{base}/chat/sessions/{TEST_SESSION_ID}/messages",
        timeout=10,
    )

    if assert_status(r, 200, f"GET /chat/sessions/{TEST_SESSION_ID}/messages"):
        data = r.json()
        assert_key(data, "messages", "Session messages response")
        messages = data.get("messages", [])
        info(f"Messages stored in session = {len(messages)}")
        for i, msg in enumerate(messages[:4], 1):  # Show at most first 4.
            role    = msg.get("role", "?")
            content = msg.get("content", "")[:80]
            info(f"  [{i}] {role}: {content}...")


def test_delete_session(base: str):
    header("TEST 8 — Delete Session  DELETE /chat/sessions/{session_id}")
    r = requests.delete(
        f"{base}/chat/sessions/{TEST_SESSION_ID}",
        timeout=10,
    )
    if assert_status(r, 200, f"DELETE /chat/sessions/{TEST_SESSION_ID}"):
        data = r.json()
        info(f"message    = {data.get('message')}")
        info(f"session_id = {data.get('session_id')}")


def test_delete_document(base: str, document_id: int):
    header(f"TEST 9 — Delete Document  DELETE /documents/{document_id}")
    r = requests.delete(f"{base}/documents/{document_id}", timeout=10)

    if assert_status(r, 200, f"DELETE /documents/{document_id}"):
        data = r.json()
        info(f"message     = {data.get('message')}")
        info(f"DOCUMENT_ID = {data.get('DOCUMENT_ID')}")


def test_delete_missing_document(base: str):
    header("TEST 10 — 404 on Unknown Document  DELETE /documents/99999")
    r = requests.delete(f"{base}/documents/99999", timeout=10)
    assert_status(r, 404, "DELETE /documents/99999 (expect 404)")


# ================================================================== #
#  Main runner                                                        #
# ================================================================== #

def main():
    global doc_id

    parser = argparse.ArgumentParser(description="End-to-end API test runner")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running API (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  PDF RAG Chat API — End-to-End Test Suite{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Target  : {base}")
    print(f"  Session : {TEST_SESSION_ID}")
    print(f"  PDF     : {SAMPLE_PDF_PATH}")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # ---- Core pipeline ----
        test_health(base)
        doc_id = test_upload(base)
        test_list_documents(base)
        test_chat_query(base)
        test_multi_turn_query(base)
        test_list_sessions(base)
        test_get_session_messages(base)

        # ---- Error path ----
        test_delete_missing_document(base)

    finally:
        # ---- Cleanup — always runs even on crash ----
        print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
        print(f"{BOLD}{CYAN}  CLEANUP{RESET}")
        print(f"{BOLD}{CYAN}{'─'*60}{RESET}")
        test_delete_session(base)
        if doc_id is not None:
            test_delete_document(base, doc_id)

    # ---- Summary ----
    total = passed + failed
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  RESULTS: {passed}/{total} tests passed{RESET}")
    if failed == 0:
        print(f"{GREEN}{BOLD}  ✔  All tests passed!{RESET}")
    else:
        print(f"{RED}{BOLD}  ✖  {failed} test(s) failed.{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
