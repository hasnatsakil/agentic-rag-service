# Agentic RAG Service

> **Production-ready, self-correcting Retrieval-Augmented Generation API** built from scratch in Python — no LangChain, no LlamaIndex for core retrieval.

**Live API Docs (Swagger UI):** [https://agentic-rag-service.onrender.com/docs](https://agentic-rag-service.onrender.com/docs)

---

## Overview

This service implements a full **Agentic RAG pipeline** that goes well beyond simple retrieve-and-read. The LLM is given tool-calling capability to decide *when* and *what* to search. Retrieved chunks are ranked twice, graded for relevance by a judge model, and the final answer is verified for hallucination — all inside a stateful LangGraph workflow. Multi-turn conversation memory is persisted to PostgreSQL with automatic summarisation compaction.

> **Zero framework bloat for core retrieval.** Chunking, cosine similarity, hybrid search, and Reciprocal Rank Fusion are implemented in pure Python and SQL — no LangChain or LlamaIndex wrappers.

---

## Key Features

| Feature | Details |
|---|---|
| **Native LLM Tool Calling** | The `agent_node` sends the OpenAI-compatible `search_pdf_database` tool schema to the LLM. The LLM decides autonomously whether to call the database or answer directly from memory. |
| **Hybrid Search (pgvector + FTS)** | Dense cosine similarity search (`pgvector` `<=>`) fused with PostgreSQL Full-Text Search (`ts_rank`, `websearch_to_tsquery`) in a single ranked list. |
| **Reciprocal Rank Fusion (RRF)** | Mathematically fuses vector and keyword rankings without hand-tuned weights: $RRF(d) = \sum_{r \in R} \frac{1}{60 + r(d)}$ |
| **Two-Pass Re-ranking** | Pass 1 — lightweight keyword overlap lexical rescoring. Pass 2 (optional) — LLM-as-a-Judge evaluates all candidates and returns a best-first ranking. |
| **Self-Correcting Graph** | LangGraph loop: retrieve → grade relevance → rewrite query on failure → select context → answer → verify factuality. |
| **Multi-Turn Chat Memory** | Per-session message history (last 20 turns) and a running summary of older turns, both persisted to Neon PostgreSQL. |
| **Automatic Summary Compaction** | After 10 messages, a background task prompts an LLM to merge the latest turn into a ≤3-sentence running summary, preventing prompt overflow. |
| **FastAPI REST Service** | Full document lifecycle (upload, list, delete) and a full chat session lifecycle (query, list sessions, delete session, view history). |
| **Observability** | Structured JSON logging via `structlog` and optional LangSmith tracing for all LLM completions. |

---

## Architecture

### Agentic LangGraph Flow (Current)

```
POST /chat/query
      │
      ▼
[agent_node] ─── has answer? ──► [check_hallucination] ──► END
      │ needs tool call
      ▼
[execute_tool_node]
      │
      ├── use_llm_rerank=True? ──► [llm_rerank_node]
      │                                    │
      ▼                                    ▼
[grade_documents_node] ◄────────────────────
      │
      ├── relevant?   ──► [select_context_node] ──► [agent_node]  (loop back)
      ├── can retry?  ──► [rewrite_query_node]  ──► [execute_tool_node]
      └── exhausted   ──► [no_context_node]     ──► END
```

**Node Summary:**

| Node | Role |
|---|---|
| `agent_node` | Calls LLM with tool schema; answers directly or requests a database search. Receives chat history + running summary. |
| `execute_tool_node` | Embeds the search query, runs hybrid search (vector + keyword, fused via RRF), applies lexical reranking (Pass 1). |
| `llm_rerank_node` | Optional Pass 2 re-ranking: an LLM judge scores all candidate chunks and returns them in best-first order. |
| `grade_documents_node` | A lightweight judge LLM grades each candidate chunk as relevant / not relevant to the user's question. |
| `rewrite_query_node` | If no relevant chunks found, an LLM rewrites the query into keyword-style search terms before retrying. |
| `select_context_node` | Trims selected chunks to fit within `MAX_CONTEXT_CHARS` and `ANSWER_K` budget before passing to the LLM. |
| `check_hallucination_node` | A verifier LLM checks if the generated answer is grounded in retrieved facts; prepends a warning if not. |
| `no_context_node` | Returns a helpful fallback message when all retries are exhausted without finding relevant context. |

---

## Directory Structure

```
test_rag/
│
├── api.py                      ← FastAPI application entry point, router registration
├── dependencies.py             ← FastAPI dependency injection (NeonVectorStore, IngestService)
├── schemas.py                  ← Pydantic request/response schemas (all API I/O contracts)
├── render.yaml                 ← Render.com deployment blueprint
├── requirements.txt            ← Python dependencies
│
├── config/
│   ├── __init__.py             ← Application settings (SEARCH_K, ANSWER_K, MIN_SCORE, etc.)
│   └── openrouter_settings.py  ← OpenRouter client, model enums, router config
│
├── core/
│   ├── chunking.py             ← RecursiveChunk: paragraph → sentence → sliding-word-window splitter
│   ├── document_loader.py      ← PDF page extraction with page-offset metadata (pypdf)
│   ├── embeddings.py           ← OpenRouterEmbeddingClient with automatic model fallback
│   ├── models.py               ← Dataclass DTOs: RetrievalResult, IngestResult, ChatResult
│   ├── rag_engine.py           ← Cosine similarity, context builder, lexical reranker, tool schema
│   ├── vector_store.py         ← NeonVectorStore: vector search, keyword search, hybrid RRF search
│   └── chat_history_store.py   ← ChatHistoryStore: per-session message history + summary persistence
│
├── graph/
│   └── rag_graph.py            ← LangGraph StateGraph: all nodes, routing functions, compiled graph
│
├── routes/
│   ├── health.py               ← GET /health — live database ping
│   ├── documents.py            ← POST /documents/upload, GET /documents, DELETE /documents/{id}
│   └── chat.py                 ← POST /chat/query, GET/DELETE /chat/sessions/*
│
├── services/
│   ├── agent_completion.py     ← agent_complete(): unified LLM call wrapper (model selection, logging)
│   ├── graph_services.py       ← GraphService: translates API request → RAGState → ChatResult
│   ├── ingest_service.py       ← IngestService: load PDF → chunk → embed → store to Neon
│   └── compaction_service.py   ← save_and_compact_workflow(): background history save + summarisation
│
├── crew/
│   ├── crews/rag_crew.py       ← CrewAI multi-agent team (Retriever, Synthesizer, Verifier) — CLI only
│   └── tools/rag_tool.py       ← Custom CrewAI search tool backed by the RAG graph
│
├── scripts/
│   ├── test_full_pipeline.py   ← Python end-to-end test runner (all endpoints)
│   └── test_api.sh             ← Bash/curl test suite (all endpoints)
│
├── data/
│   └── sample.pdf              ← Sample PDF for local testing
│
└── docs/
    └── project_flow.md         ← Extended Mermaid flow diagrams
```

---

## Database Schema

```sql
-- Document registry
CREATE TABLE documents (
    id         SERIAL PRIMARY KEY,
    file_name  TEXT NOT NULL,
    summary    TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector store — one row per text chunk
CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   VECTOR(1536),
    page_number INTEGER
);

-- Multi-turn chat history
CREATE TABLE chat_messages (
    id         SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,   -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    metadata   JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Running conversation summaries (one row per session)
CREATE TABLE chat_summaries (
    session_id   TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Pydantic v2 |
| **Agentic Workflow** | LangGraph (`StateGraph`) |
| **LLM API** | OpenRouter (`openrouter.ai`) via OpenAI SDK |
| **Embeddings** | `openai/text-embedding-3-small` via OpenRouter |
| **Vector Database** | Neon Serverless PostgreSQL + `pgvector` |
| **Vector Search** | `pgvector` cosine distance (`<=>`) |
| **Keyword Search** | PostgreSQL Full-Text Search (`ts_rank`, `websearch_to_tsquery`) |
| **PDF Extraction** | `pypdf` |
| **Database Driver** | `psycopg` v3 (psycopg3) |
| **Logging** | `structlog` (structured JSON) |
| **Observability** | LangSmith (optional tracing) |
| **Multi-Agent CLI** | CrewAI |
| **Deployment** | Render Web Service (`render.yaml`) |

---

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/hasnatsakil/agentic-rag-service.git
cd agentic-rag-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file at the project root:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_api_key
DATABASE_URL=postgresql://user:password@ep-cool-db.region.aws.neon.tech/dbname

# Optional — LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agentic-rag-service
```

### 3. Run the API Server

```bash
uvicorn api:app --reload
```

Browse the interactive Swagger UI at **`http://127.0.0.1:8000/docs`**.

---

## Testing

Two test scripts in `scripts/` validate the full pipeline from the terminal. Both test every endpoint in order, verify response payloads, and clean up after themselves.

Start the API server first:
```bash
uvicorn api:app --reload
```

### Option A — Python runner (recommended)

```bash
python scripts/test_full_pipeline.py
python scripts/test_full_pipeline.py --base-url https://your-service.onrender.com
```

### Option B — Bash/curl suite

```bash
bash scripts/test_api.sh
bash scripts/test_api.sh https://your-service.onrender.com
```

### What both scripts verify

| Step | Endpoint | Verified |
|---|---|---|
| 1 | `GET /health` | Server up, DB connected |
| 2 | `POST /documents/upload` | PDF ingested, document ID returned |
| 3 | `GET /documents` | Document appears in list |
| 4 | `POST /chat/query` (turn 1) | Answer, sources, debug metrics |
| 5 | `POST /chat/query` (turn 2) | Session memory carries over |
| 6 | `GET /chat/sessions` | Test session visible |
| 7 | `GET /chat/sessions/{id}/messages` | 4 messages persisted |
| 8 | `DELETE /documents/99999` | 404 for unknown document |
| C | `DELETE /chat/sessions/{id}` | Session cleaned up |
| C | `DELETE /documents/{id}` | Document cleaned up |

---

## API Reference

### Document Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a PDF; extract, chunk, embed, and store all pages. |
| `GET` | `/documents` | List all indexed documents (id, filename, created_at). |
| `DELETE` | `/documents/{id}` | Remove a document and all its vector chunks (cascade). |

### Chat Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat/query` | Submit a question; get an LLM answer with source citations and debug metrics. |
| `GET` | `/chat/sessions` | List all chat sessions with their latest activity timestamp. |
| `GET` | `/chat/sessions/{session_id}/messages` | Retrieve the full chronological message log of a session. |
| `DELETE` | `/chat/sessions/{session_id}` | Delete all messages and summary for a session. |

### Other

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — pings the database and returns server status. |

---

### Chat Query Request Body (`POST /chat/query`)

```json
{
  "session_id": "my-chat-thread-1",
  "question": "What are the visa requirements for international students?",
  "SEARCH_K": 15,
  "GRADE_K": 6,
  "ANSWER_K": 3,
  "MIN_SCORE": 0.0,
  "MAX_CONTEXT_CHARS": 3000,
  "use_llm_rerank": false
}
```

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Unique session string. History and summary are fetched/saved against this key. |
| `question` | `str` | The user's natural-language question. |
| `SEARCH_K` | `int` | Candidate chunks fetched from the vector store per search attempt. |
| `GRADE_K` | `int` | How many top candidates the relevance grader will evaluate. |
| `ANSWER_K` | `int` | Maximum chunks forwarded to the LLM for answer generation. |
| `MIN_SCORE` | `float` | Retrieval score threshold; chunks below this are discarded. |
| `MAX_CONTEXT_CHARS` | `int` | Hard cap on total context characters sent to the LLM. |
| `use_llm_rerank` | `bool` | Enable Pass 2 LLM-as-a-Judge re-ranking before grading. |

### Chat Query Response Body

```json
{
  "answer": "International students require a Single Entry Visa...",
  "sources": [
    {
      "label": "Page 3, Chunk 7",
      "score": 0.048,
      "chunk_id": 6,
      "page_number": 3,
      "chunk_text": "Malaysia removed quarantine for fully vaccinated..."
    }
  ],
  "process_time_ms": 4821.5,
  "debug": {
    "used_rewrite": false,
    "is_grounded": true,
    "retrieval_count": 12,
    "selected_count": 3
  }
}
```

---

## Data Flow Diagrams

### 1. PDF Ingestion Flow

```mermaid
flowchart TD
    A["POST /documents/upload"] --> B["routes/documents.py"]
    B --> C["IngestService.ingest_pdf()"]

    C --> D["DocumentLoader.load_pdf_pages()"]
    D --> E["list of {page_number, text}"]

    E --> F["RecursiveChunk.chunk() per page"]
    F --> F1{"Paragraph fits ≤ max_words?"}
    F1 -- Yes --> F2["Keep paragraph as chunk"]
    F1 -- No --> F3{"Sentence fits?"}
    F3 -- Yes --> F4["Keep sentence as chunk"]
    F3 -- No --> F5["chunk_by_words(sliding window, overlap)"]

    F2 & F4 & F5 --> G["all_chunks + page_numbers"]

    G --> H["RAGEngine.embed_chunks()"]
    H --> H1["OpenRouterEmbeddingClient (auto-fallback)"]
    H1 --> H2["OpenRouter API → embedding vectors [1536d]"]

    H2 --> I["NeonVectorStore.create_document()"]
    I --> I1["INSERT INTO documents → document_id"]
    I1 --> J["NeonVectorStore.insert_chunks()"]
    J --> J1["INSERT INTO document_chunks (chunk_text, embedding, page_number)"]
    J1 --> K["Returns IngestResult (id, file_name, page_count, chunk_count)"]

    style A fill:#2a9d8f,color:#fff
    style K fill:#1b4332,color:#fff
    style H2 fill:#e76f51,color:#fff
```

### 2. Agentic Chat Query Flow

```mermaid
flowchart TD
    A["POST /chat/query"] --> B["routes/chat.py"]
    B --> B1["ChatHistoryStore.get_last_20_message()"]
    B --> B2["ChatHistoryStore.get_summary()"]
    B1 & B2 --> C["GraphService.ask_pdf_with_graph()"]
    C --> D["rag_graph.invoke(RAGState)"]

    D --> AGENT["agent_node: LLM + tool schema\n(receives history + summary + doc catalog)"]
    AGENT -- "tool call" --> TOOL["execute_tool_node:\nhybrid_search + Pass 1 rerank"]
    AGENT -- "direct answer" --> HALL["check_hallucination_node"]

    TOOL --> OPT{"use_llm_rerank?"}
    OPT -- Yes --> LLM_R["llm_rerank_node:\nLLM judge reorders chunks"]
    OPT -- No --> GRADE
    LLM_R --> GRADE["grade_documents_node:\njudge each chunk yes/no"]

    GRADE -- "relevant" --> SEL["select_context_node:\ntrim to ANSWER_K / MAX_CONTEXT_CHARS"]
    GRADE -- "retry budget left" --> REW["rewrite_query_node:\nLLM rewrites query"]
    GRADE -- "exhausted" --> NO["no_context_node"]

    REW --> TOOL
    SEL --> AGENT
    AGENT -- "now has context, answers" --> HALL
    HALL --> END["END: ChatResult"]
    NO --> END

    END --> E["BackgroundTask: save_and_compact_workflow()"]
    E --> E1["ChatHistoryStore.save_message() ×2"]
    E --> E2["LLM updates running summary → ChatHistoryStore.update_summary()"]
    END --> F["Returns QueryResponse JSON"]

    style A fill:#2a9d8f,color:#fff
    style END fill:#1b4332,color:#fff
    style HALL fill:#e76f51,color:#fff
    style GRADE fill:#e76f51,color:#fff
    style LLM_R fill:#e76f51,color:#fff
    style REW fill:#e76f51,color:#fff
```

### 3. Hybrid Search & Re-ranking Detail

```mermaid
flowchart TD
    Q["search_question (from agent tool call)"] --> EMB["embed_query()"]
    EMB --> VS["NeonVectorStore.similarity_search()\npgvector cosine distance <=>"]
    Q --> KS["NeonVectorStore.keyword_search()\nts_rank + websearch_to_tsquery"]
    VS & KS --> RRF["Reciprocal Rank Fusion\nRRF(d) = Σ 1/(60 + rank)"]
    RRF --> P1["RAGEngine.rerank_results()\nPass 1: keyword overlap boost"]
    P1 --> FILTER["Filter by MIN_SCORE"]
    FILTER --> OPT{"use_llm_rerank?"}
    OPT -- Yes --> P2["llm_rerank_node\nPass 2: LLM judge best-first ordering"]
    OPT -- No --> GRADE_IN["grade_documents_node"]
    P2 --> GRADE_IN

    style RRF fill:#264653,color:#fff
    style P1 fill:#457b9d,color:#fff
    style P2 fill:#e76f51,color:#fff
```

### 4. Multi-Turn Memory Architecture

```mermaid
flowchart TD
    subgraph Per Request
        R["POST /chat/query\n(session_id, question)"] --> HS["ChatHistoryStore:\nget_last_20_message()"]
        R --> SS["ChatHistoryStore:\nget_summary()"]
        HS & SS --> GS["GraphService → RAGState\n(history: list[dict], summary: str)"]
        GS --> AG["agent_node receives:\n• history (last 20 turns)\n• summary (older turns)\n• doc catalog"]
    end

    subgraph Background after response
        BG["save_and_compact_workflow()"] --> SM["save_message(user + assistant)"]
        BG --> CHK{"history count >= 10?"}
        CHK -- Yes --> LLM_S["LLM: merge latest turn\ninto running summary (≤3 sentences)"]
        LLM_S --> US["update_summary(session_id, new_summary)"]
        CHK -- No --> SKIP["Skip summary update"]
    end

    style AG fill:#264653,color:#fff
    style LLM_S fill:#e76f51,color:#fff
```

---

## Why LangGraph? The Self-Correcting Architecture

Standard RAG systems blindly retrieve documents and pass them to the LLM, producing hallucinations when the context is poor. This service uses a stateful graph where:

1. **Native Tool Calling** — The LLM autonomously decides whether to query the database or answer from memory/history. It selects the correct document ID from the catalog on its own.
2. **LLM Grading** — A judge model evaluates each retrieved chunk independently and filters out irrelevant content before the answer model sees it.
3. **Self-Correction** — If grading fails, an LLM rewrites the query into keyword-style search terms and retries retrieval automatically.
4. **Hallucination Prevention** — A verifier model cross-references the final answer against the retrieved facts. An ungrounded answer gets a prepended warning rather than being silently returned.
5. **Persistent Memory** — Session history and compact summaries are stored in PostgreSQL, so the agent remembers past interactions across API requests.

---

## Deployment (Render)

The repository includes a `render.yaml` blueprint for one-click deployment:

1. Push to GitHub.
2. Log into [Render](https://render.com) → **New → Blueprint**.
3. Connect the repository; Render reads `render.yaml` automatically.
4. Add secret environment variables in the Render dashboard:
   - `OPENROUTER_API_KEY`
   - `DATABASE_URL`
   - `LANGSMITH_API_KEY` *(optional)*
5. Click **Deploy**.

---

## License

MIT
