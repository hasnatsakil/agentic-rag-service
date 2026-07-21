#!/usr/bin/env bash
# =============================================================================
#  scripts/test_api.sh
#  Full end-to-end curl-based test suite for the PDF RAG Chat API.
#
#  Tests every REST endpoint in the correct order with clear pass/fail output.
#  Cleanup (session + document deletion) always runs, even on failure.
#
#  Prerequisites:
#    - curl  (standard on Linux/macOS)
#    - jq    (apt install jq  /  brew install jq)   <- optional but recommended
#    - API server running:  uvicorn api:app --reload
#    - Sample PDF present:  data/sample.pdf
#
#  Usage:
#    bash scripts/test_api.sh
#    bash scripts/test_api.sh http://your-render-url.com
# =============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
BASE_URL="${1:-http://127.0.0.1:8000}"
PDF_PATH="data/sample.pdf"
SESSION_ID="test-session-$(date +%s)"
QUESTION="What is this document about?"

# ---- Colours ----------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

ok()     { echo -e "  ${GREEN}✔  $*${RESET}"; }
fail()   { echo -e "  ${RED}✖  $*${RESET}"; }
info()   { echo -e "  ${CYAN}→  $*${RESET}"; }
warn()   { echo -e "  ${YELLOW}⚠  $*${RESET}"; }
header() { echo -e "\n${BOLD}${CYAN}────────────────────────────────────────────────────────────${RESET}";
           echo -e "${BOLD}${CYAN}  $*${RESET}";
           echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────${RESET}"; }

# ---- Helpers ----------------------------------------------------------------
PASS=0
FAIL=0
DOC_ID=""

# Check if jq is available for pretty-printing.
HAS_JQ=false
command -v jq &>/dev/null && HAS_JQ=true

pretty() {
    # Pretty-print JSON if jq is available, else raw output.
    if $HAS_JQ; then
        echo "$1" | jq .
    else
        echo "$1"
    fi
}

assert_status() {
    # Usage: assert_status <actual_code> <expected_code> <test_label>
    if [ "$1" -eq "$2" ]; then
        ok "$3 — HTTP $1"
        ((PASS++)) || true
        return 0
    else
        fail "$3 — Expected HTTP $2, got HTTP $1"
        ((FAIL++)) || true
        return 1
    fi
}

# ============================================================================
#  Print header
# ============================================================================
echo -e "\n${BOLD}════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  PDF RAG Chat API — Shell Test Suite${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
echo -e "  Target  : ${BASE_URL}"
echo -e "  Session : ${SESSION_ID}"
echo -e "  PDF     : ${PDF_PATH}"
echo -e "  Started : $(date '+%Y-%m-%d %H:%M:%S')"

# ============================================================================
#  TEST 1 — Health Check
# ============================================================================
header "TEST 1 — Health Check  GET /health"

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    "${BASE_URL}/health")
BODY=$(cat /tmp/rag_resp.json)

assert_status "$HTTP_CODE" 200 "GET /health" || true
STATUS_VAL=$(echo "$BODY" | $HAS_JQ && jq -r '.status' <<< "$BODY" || echo "?")
DB_VAL=$(echo "$BODY" | $HAS_JQ && jq -r '.database' <<< "$BODY" || echo "?")
info "status   = ${STATUS_VAL}"
info "database = ${DB_VAL}"

if [ "$STATUS_VAL" != "ok" ]; then
    warn "Database may not be healthy — subsequent tests may fail."
fi

# ============================================================================
#  TEST 2 — Upload PDF
# ============================================================================
header "TEST 2 — PDF Upload  POST /documents/upload"

if [ ! -f "$PDF_PATH" ]; then
    fail "PDF not found at '${PDF_PATH}'. Skipping upload."
    warn "Cannot run chat tests without a document. Exiting."
    exit 1
fi

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    -F "file=@${PDF_PATH};type=application/pdf" \
    "${BASE_URL}/documents/upload")
BODY=$(cat /tmp/rag_resp.json)

if assert_status "$HTTP_CODE" 200 "POST /documents/upload"; then
    DOC_ID=$(echo "$BODY" | $HAS_JQ && jq -r '.DOCUMENT_ID' <<< "$BODY" || echo "")
    info "DOCUMENT_ID = ${DOC_ID}"
    info "file_name   = $(echo "$BODY" | $HAS_JQ && jq -r '.file_name' <<< "$BODY" || echo '?')"
    info "page_count  = $(echo "$BODY" | $HAS_JQ && jq -r '.page_count' <<< "$BODY" || echo '?')"
    info "chunk_count = $(echo "$BODY" | $HAS_JQ && jq -r '.chunk_count' <<< "$BODY" || echo '?')"
fi

# ============================================================================
#  TEST 3 — List Documents
# ============================================================================
header "TEST 3 — List Documents  GET /documents"

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    "${BASE_URL}/documents")
BODY=$(cat /tmp/rag_resp.json)

if assert_status "$HTTP_CODE" 200 "GET /documents"; then
    COUNT=$(echo "$BODY" | $HAS_JQ && jq '.documents | length' <<< "$BODY" || echo "?")
    info "Total documents = ${COUNT}"
fi

# ============================================================================
#  TEST 4 — Chat Query (turn 1)
# ============================================================================
header "TEST 4 — Chat Query  POST /chat/query"
info "session_id = ${SESSION_ID}"
info "question   = ${QUESTION}"

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "{
        \"session_id\": \"${SESSION_ID}\",
        \"question\": \"${QUESTION}\",
        \"SEARCH_K\": 15,
        \"GRADE_K\": 6,
        \"ANSWER_K\": 3,
        \"MIN_SCORE\": 0.0,
        \"MAX_CONTEXT_CHARS\": 3000,
        \"use_llm_rerank\": false
    }" \
    "${BASE_URL}/chat/query")
BODY=$(cat /tmp/rag_resp.json)

if assert_status "$HTTP_CODE" 200 "POST /chat/query"; then
    ANSWER=$(echo "$BODY" | $HAS_JQ && jq -r '.answer' <<< "$BODY" | head -c 120 || echo "?")
    SRC_CNT=$(echo "$BODY" | $HAS_JQ && jq '.sources | length' <<< "$BODY" || echo "?")
    GROUNDED=$(echo "$BODY" | $HAS_JQ && jq -r '.debug.is_grounded' <<< "$BODY" || echo "?")
    REWRITE=$(echo "$BODY" | $HAS_JQ && jq -r '.debug.used_rewrite' <<< "$BODY" || echo "?")
    PROC_MS=$(echo "$BODY" | $HAS_JQ && jq -r '.process_time_ms' <<< "$BODY" || echo "?")
    info "answer (120 chars) = ${ANSWER}..."
    info "sources returned   = ${SRC_CNT}"
    info "is_grounded        = ${GROUNDED}"
    info "used_rewrite       = ${REWRITE}"
    info "process_time_ms    = ${PROC_MS} ms"
fi

# ============================================================================
#  TEST 5 — Multi-Turn Memory (turn 2, same session)
# ============================================================================
header "TEST 5 — Multi-Turn Memory  POST /chat/query (turn 2)"
info "Same session_id — agent should recall previous turn"

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "{
        \"session_id\": \"${SESSION_ID}\",
        \"question\": \"Can you summarise what you just told me?\",
        \"SEARCH_K\": 15,
        \"GRADE_K\": 6,
        \"ANSWER_K\": 3,
        \"MIN_SCORE\": 0.0,
        \"MAX_CONTEXT_CHARS\": 3000,
        \"use_llm_rerank\": false
    }" \
    "${BASE_URL}/chat/query")
BODY=$(cat /tmp/rag_resp.json)

if assert_status "$HTTP_CODE" 200 "POST /chat/query (turn 2)"; then
    ANSWER=$(echo "$BODY" | $HAS_JQ && jq -r '.answer' <<< "$BODY" | head -c 120 || echo "?")
    PROC_MS=$(echo "$BODY" | $HAS_JQ && jq -r '.process_time_ms' <<< "$BODY" || echo "?")
    info "answer (120 chars) = ${ANSWER}..."
    info "process_time_ms    = ${PROC_MS} ms"
fi

# ============================================================================
#  TEST 6 — List Sessions
# ============================================================================
header "TEST 6 — List Sessions  GET /chat/sessions"

sleep 1   # Brief pause for background compaction task.

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    "${BASE_URL}/chat/sessions")
BODY=$(cat /tmp/rag_resp.json)

if assert_status "$HTTP_CODE" 200 "GET /chat/sessions"; then
    COUNT=$(echo "$BODY" | $HAS_JQ && jq '.sessions | length' <<< "$BODY" || echo "?")
    info "Active sessions = ${COUNT}"
fi

# ============================================================================
#  TEST 7 — Get Session Messages
# ============================================================================
header "TEST 7 — Session Messages  GET /chat/sessions/${SESSION_ID}/messages"

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    "${BASE_URL}/chat/sessions/${SESSION_ID}/messages")
BODY=$(cat /tmp/rag_resp.json)

if assert_status "$HTTP_CODE" 200 "GET /chat/sessions/${SESSION_ID}/messages"; then
    COUNT=$(echo "$BODY" | $HAS_JQ && jq '.messages | length' <<< "$BODY" || echo "?")
    info "Messages stored = ${COUNT} (expect 4 — 2 turns × user+assistant)"
fi

# ============================================================================
#  TEST 8 — 404 on Unknown Document
# ============================================================================
header "TEST 8 — 404 on Unknown Document  DELETE /documents/99999"

HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    -X DELETE "${BASE_URL}/documents/99999")

assert_status "$HTTP_CODE" 404 "DELETE /documents/99999 (expect 404)" || true

# ============================================================================
#  CLEANUP — always runs
# ============================================================================
header "CLEANUP"

# Delete test session.
info "Deleting session: ${SESSION_ID}"
HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
    -X DELETE "${BASE_URL}/chat/sessions/${SESSION_ID}")
assert_status "$HTTP_CODE" 200 "DELETE /chat/sessions/${SESSION_ID}" || true

# Delete uploaded document if we have its ID.
if [ -n "$DOC_ID" ] && [ "$DOC_ID" != "null" ]; then
    info "Deleting document ID: ${DOC_ID}"
    HTTP_CODE=$(curl -s -o /tmp/rag_resp.json -w "%{http_code}" \
        -X DELETE "${BASE_URL}/documents/${DOC_ID}")
    assert_status "$HTTP_CODE" 200 "DELETE /documents/${DOC_ID}" || true
else
    warn "No document ID recorded — skipping document cleanup."
fi

# ============================================================================
#  Summary
# ============================================================================
TOTAL=$((PASS + FAIL))
echo -e "\n${BOLD}════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  RESULTS: ${PASS}/${TOTAL} tests passed${RESET}"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ✔  All tests passed!${RESET}"
else
    echo -e "${RED}${BOLD}  ✖  ${FAIL} test(s) failed.${RESET}"
fi
echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}\n"

exit $FAIL
