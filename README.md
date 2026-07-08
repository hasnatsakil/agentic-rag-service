# 🤖 Agentic RAG Service — Custom RAG with FastAPI, CrewAI (CLI Only), & LangGraph

## 🚀 Live Demo
* **API Docs / Swagger UI:** [https://agentic-rag-service.onrender.com/docs](https://agentic-rag-service.onrender.com/docs)

---

This repository contains a production-ready, modular **Agentic RAG Service** built from scratch in Python. It implements a self-correcting RAG pipeline that moves from relational databases and serverless Postgres to stateful routing graphs and multi-agent validation.

> **Zero Framework Bloat for Core Retrieval**: The core text processing, custom recursive chunking, pgvector similarity search, and SQL-native full-text keyword queries are written in pure Python/SQL without using LangChain or LlamaIndex wrappers.

---

## 🌟 Key Features

*   **Database-Native Hybrid Search**: Combines semantic vector similarity search (`pgvector` cosine distance `<=>`) with exact keyword search (Postgres Full-Text Search using `ts_rank` and `websearch_to_tsquery`) directly inside SQL.
*   **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense search result rankings mathematically without needing to manually tune weights:
    $$RRF(d) = \sum_{r \in R} \frac{1}{60 + r(d)}$$
*   **Two-Pass Re-ranking**:
    *   **Pass 1 (Lexical Rescoring)**: A lightweight keyword overlap metric that boosts chunks with exact query matches.
    *   **Pass 2 (Stateful LLM Reranking)**: An optional LangGraph node that uses a lightweight LLM-as-a-Judge to evaluate retrieved chunks and bubble up the most relevant ones.
*   **Self-Correcting Graph Workflows (LangGraph)**: Stateful loop that retrieves context, grades relevance via an LLM, automatically rewrites queries on low-quality search results, generates answers, and verifies factuality (hallucination check) before returning the response.
*   **FastAPI REST Web Service**: Endpoints for dynamic PDF document uploading, listing, deletion, and context-retrieval chat.
*   **CrewAI Collaborative CLI**: A cooperative team of agents (Retriever, Synthesizer, and Verifier) executing QA workflows in the terminal.
*   **Observability (LangSmith)**: Native tracing integration for all agentic chains.

---

## ⚙️ Directory Structure

```
test_rag/
│
├── api.py                    ← FastAPI Web Application Entry Point
├── dependencies.py           ← FastAPI dependency injection mapping
├── schemas.py                ← Pydantic API validation schemas
├── render.yaml               ← Deployment config for render.com Web Service
├── requirements.txt          ← System dependencies (local & Render)
│
├── config/
│   ├── __init__.py           ← Settings / configuration loader
│   └── openrouter_settings.py← OpenRouter LLM and embed models configuration
│
├── routes/
│   ├── health.py             ← Health status check with live database ping
│   ├── documents.py          ← File uploading, listing, and deletion endpoints
│   └── chat.py               ← Querying and generating answers via LangGraph
│
├── services/
│   ├── __init__.py           
│   ├── agent_completion.py   ← OpenRouter API completion and grading caller
│   ├── graph_services.py     ← Service wrapper for query execution via LangGraph
│   └── ingest_service.py     ← PDF loading, chunking, embedding & vector DB insertion
│
├── core/
│   ├── __init__.py           
│   ├── chunking.py           ← Recursive paragraph/sentence/word chunker
│   ├── document_loader.py    ← pypdf loader with page offset mapping
│   ├── embeddings.py         ← OpenRouter embedding client
│   ├── models.py             ← Return structures (RetrievalResult, ChatResult, etc.)
│   ├── rag_engine.py         ← Prompt builders, RAG context compilers, and lexical reranker
│   └── vector_store.py       ← Neon PostgreSQL connection and FTS + pgvector hybrid search
│
├── crew/
│   ├── __init__.py           
│   ├── crews/
│   │   ├── __init__.py       
│   │   └── rag_crew.py       ← Multi-agent RAG workflow (Retriever, Synthesizer, Verifier)
│   └── tools/
│       ├── __init__.py       
│       └── rag_tool.py       ← Custom RAG search tool utilizing the database
│
├── graph/
│   ├── __init__.py           
│   └── rag_graph.py          ← LangGraph StateGraph routing nodes (retrieve, rerank, grading, answer)
│
├── scripts/
│   ├── crew_runner.py        ← Script to kickoff the CrewAI RAG agent
│   ├── compare_retrieval.py  ← Script to compare Keyword vs Vector vs Hybrid (RRF) search
│   ├── test_hybrid_search.py ← Script to smoke-test RRF retrieval
│   └── test_keyword_search.py← Script to smoke-test PostgreSQL keyword search
│
├── data/
│   └── sample.pdf            ← PDF file used for testing
│
└── docs/
    └── project_flow.md       ← Full Mermaid diagrams
```

---

## 📸 Architectural Visuals

Detailed diagrams showing how data flows through the application (see [docs/project_flow.md](docs/project_flow.md) for full-res versions):

### 1. Ingesting a PDF (API Flow)
```mermaid
flowchart TD
    A["POST /documents/upload"] --> B["routes/documents.py (upload_file)"]
    B --> C["IngestService.ingest_pdf()"]
    
    C --> D["DocumentLoader.load_pdf_pages()"]
    D --> E["list of {page_number, text}"]
    
    E --> F["RecursiveChunk.chunk() per page"]
    F --> F1{"Paragraph fits?"}
    F1 -- Yes --> F2["Keep as chunk"]
    F1 -- No --> F3{"Sentence fits?"}
    F3 -- Yes --> F4["Keep as chunk"]
    F3 -- No --> F5["chunk_by_words(max_words=40, overlap=5)"]
    
    F2 & F4 & F5 --> G["all_chunks + page_numbers"]
    
    G --> H["RAGEngine.embed_chunks()"]
    H --> H1["OpenRouterEmbeddingClient.embed_documents()"]
    H1 --> H2["OpenRouter API → embedding vectors"]
    
    H2 --> I["NeonVectorStore.create_document()"]
    I --> I1["INSERT INTO documents → DOCUMENT_ID"]
    
    I1 --> J["NeonVectorStore.insert_chunks()"]
    J --> J1["INSERT INTO document_chunks
(DOCUMENT_ID, chunk_index, chunk_text, embedding, page_number)"]
    
    J1 --> K["Returns IngestResult
(DOCUMENT_ID, file_name, page_count, chunk_count)"]

    style A fill:#2a9d8f,color:#fff
    style B fill:#264653,color:#fff
    style K fill:#1b4332,color:#fff
    style H2 fill:#e76f51,color:#fff
    style I1 fill:#264653,color:#fff
    style J1 fill:#264653,color:#fff
```

### 2. Chatting with a PDF (API Query Flow via LangGraph)
```mermaid
flowchart TD
    A["POST /chat/query"] --> B["routes/chat.py (query_pdf_graph)"]
    B --> C["GraphService.ask_pdf_with_graph()"]
    
    C --> D["rag_graph.invoke()"]
    D --> E["Retrieve Node → Grade Node → Answer Node"]
    E --> J["Returns ChatResult
(answer, sources)"]
    J --> K["routes/chat.py returns QueryResponse JSON"]

    style A fill:#2a9d8f,color:#fff
    style B fill:#264653,color:#fff
    style D fill:#e76f51,color:#fff
    style E fill:#264653,color:#fff
    style J fill:#1b4332,color:#fff
```

### 3. Web API Routing & Services
```mermaid
flowchart TD
    subgraph Client["HTTP Clients (Postman/Curl/Frontend)"]
        C1["POST /documents/upload"]
        C2["POST /chat/query"]
    end

    subgraph API["FastAPI Layer"]
        R_DOC["routes/documents.py (upload_file)"]
        R_CHAT["routes/chat.py (query_pdf_graph)"]
    end

    subgraph DI["Dependency Injection"]
        D1["dependencies.py"]
    end

    subgraph Services["Service Layer"]
        IS["IngestService.ingest_pdf()"]
        GS["GraphService.ask_pdf_with_graph()"]
    end

    C1 --> R_DOC
    C2 --> R_CHAT

    R_DOC -.->|Depends| D1
    
    D1 -->|Injects| IS

    IS -->|Returns IngestResult| R_DOC
    R_CHAT -->|Calls directly| GS
    GS -->|Returns ChatResult| R_CHAT

    R_DOC -->|Returns UploadResponse| C1
    R_CHAT -->|Returns QueryResponse| C2

    style C1 fill:#2a9d8f,color:#fff
    style C2 fill:#2a9d8f,color:#fff
    style R_DOC fill:#264653,color:#fff
    style R_CHAT fill:#264653,color:#fff
    style IS fill:#e76f51,color:#fff
    style GS fill:#e76f51,color:#fff
```

### 4. CrewAI Collaborative Team Flow
```mermaid
flowchart TD
    A["scripts/crew_runner.py"] --> B["rag_crew.kickoff()"]

    subgraph Crew["CrewAI Team Orcherstration"]
        AG1["Document Retriever Agent"]
        AG2["Answer Synthesizer Agent"]
        AG3["Answer Verifier Agent"]
        
        T1["Task 1: Search PDF via custom RAG Tool"]
        T2["Task 2: Grounded QA Synthesis"]
        T3["Task 3: Fact Verification & Correction"]
    end

    B --> T1
    AG1 -->|Executes| T1
    T1 -->|Uses custom search_pdf tool| TL["crew/tools/rag_tool.py"]
    TL -->|Queries| GS["GraphService.ask_pdf_with_graph()"]
    
    T1 -->|Outputs Chunks| T2
    AG2 -->|Executes| T2
    
    T2 -->|Outputs Draft Answer| T3
    AG3 -->|Executes| T3
    T3 -->|Context checks| T1
    
    T3 -->|Returns Grounded Answer| C["Final Output Answer"]

    style A fill:#2d6a4f,color:#fff
    style GS fill:#e76f51,color:#fff
    style C fill:#1b4332,color:#fff
    style TL fill:#264653,color:#fff
```

### 5. LangGraph Self-Correcting Flow (Level 9)
```mermaid
flowchart TD
    A["graph/rag_graph.py (invoke)"] --> START["START"]

    START --> RET["retrieve_node"]

    RET --> GRADE["grade_documents_node
(LLM grades context)"]
    GRADE --> COND1{"route_after_grading"}

    COND1 -->|has_context| SEL["select_context_node"]
    COND1 -->|retry_count < max| REW["rewrite_query_node
(Expands search)"]
    COND1 -->|no retries left| NO_CTX["no_context_node"]

    REW --> RET

    SEL --> COND2{"route_after_select"}
    COND2 -->|has_context| ANS["answer_node
(LLM drafts answer)"]
    COND2 -->|no_context| NO_CTX

    ANS --> HALLUC["check_hallucination_node
(LLM verifies factuality)"]

    HALLUC --> END1["END (Return: Verified Answer)"]
    NO_CTX --> END2["END (Return: No relevant info)"]

    style A fill:#2d6a4f,color:#fff
    style GRADE fill:#e76f51,color:#fff
    style REW fill:#e76f51,color:#fff
    style HALLUC fill:#e76f51,color:#fff
    style END1 fill:#1b4332,color:#fff
    style END2 fill:#9b2226,color:#fff
```

### 🧠 Why LangGraph? (The Self-Correcting Architecture)
Standard RAG systems blindly retrieve documents and pass them to an LLM, leading to hallucinations if the context is poor. By using **LangGraph**, this service acts autonomously:
1. **Grading**: It explicitly asks a lightweight LLM-as-a-Judge if the retrieved chunks actually answer the question.
2. **Self-Correction**: If the grade is poor, it automatically rewrites the query and tries again.
3. **Hallucination Prevention**: Before returning the final answer, a strict "Verifier" LLM cross-references the answer against the retrieved chunks. If it detects a hallucination, it flags it.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (with Pydantic schemas) |
| **Agentic Frameworks**| CrewAI & LangGraph |
| **LLM API** | OpenRouter (`openrouter.ai`) |
| **Embeddings** | `openai/text-embedding-3-small` via OpenRouter |
| **Vector DB** | Neon Serverless PostgreSQL |
| **Vector Search** | `pgvector` (Cosine Similarity `<=>`) |
| **Sparse Keyword Search**| PostgreSQL Full-Text Search (`ts_rank`, `websearch_to_tsquery`) |
| **PDF Ingestion** | `pypdf` |
| **Database Driver** | `psycopg` (v3) |
| **Hosting Platform** | Render |

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/hasnatsakil/agentic-rag-service.git
cd agentic-rag-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Neon Vector database (PostgreSQL + pgvector)
DATABASE_URL=postgresql://user:password@ep-cool-db.region.aws.neon.tech/dbname

# Optional: LangSmith Tracing & Observability
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agentic-rag-service
```

---

## 💻 Running the Services

### Option A: Start the FastAPI Web Server
```bash
uvicorn api:app --reload
```

### Option B: Run the CrewAI Multi-Agent RAG Runner
```bash
python scripts/crew_runner.py
```

### Option C: Run the LangGraph Stateful Pipeline (CLI Interface)
```bash
python graph/rag_graph.py
```

---

## 🔌 API Reference

### 1. Interactive Testing (Swagger UI)
Once the FastAPI server is running, open your browser and go to:
👉 **`http://127.0.0.1:8000/docs`**

Here you can click **"Try it out"** to upload files, list documents, and send queries directly from the UI.

### 2. Programmatic Integration (Usage Example)
You can call the endpoints from any frontend (React, Flutter) or script using standard HTTP requests. 

**Usage example (Python `requests`):**
```python
import requests

BASE_URL = "https://your-render-url.com" # or http://127.0.0.1:8000

# 1. Upload a PDF
with open("sample.pdf", "rb") as f:
    upload_res = requests.post(f"{BASE_URL}/documents/upload", files={"file": f})
    
DOCUMENT_ID = upload_res.json()["document_id"]
print(f"Uploaded successfully. Document ID: {DOCUMENT_ID}")

# 2. Ask a question about the PDF
payload = {
    "DOCUMENT_ID": DOCUMENT_ID,
    "question": "What is the main conclusion of this document?",
    "use_llm_rerank": True  # Enable Pass 2 LLM Re-ranking!
}
chat_res = requests.post(f"{BASE_URL}/chat/query", json=payload)

print("
--- Answer ---")
print(chat_res.json()["answer"])
print(f"Latency: {chat_res.json()['process_time_ms']} ms")
```

### Available Endpoints
* **`POST /documents/upload`** — Uploads and vectorizes a `.pdf` file.
* **`GET /documents`** — Lists all indexed documents.
* **`DELETE /documents/{DOCUMENT_ID}`** — Removes a document and its vectors.
* **`POST /chat/query`** — Performs hybrid similarity search, optional re-ranking, and generates an LLM answer.
* **`GET /health`** — Pinger endpoint to check server status.

---

## ☁️ Deploying on Render

This repository is pre-configured for deployment as a Render **Web Service** using the included `render.yaml` configuration.

### Deploy Steps:
1. Push your code to your GitHub repository.
2. Log into [Render](https://render.com).
3. Click **New** -> **Blueprint**.
4. Connect this repository. Render will automatically read the `render.yaml` file.
5. In the Render Dashboard, fill in your Secret Environment Variables:
   - `OPENROUTER_API_KEY`
   - `DATABASE_URL`
   - `LANGSMITH_API_KEY` (optional)
6. Click **Deploy**. The API service will build and start up automatically on Render.

---

## 📜 License

MIT
