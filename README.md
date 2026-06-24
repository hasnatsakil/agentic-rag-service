## 🚀 Live Demo
API Docs: https://agentic-rag-service.onrender.com/docs

# 🤖 Agentic RAG Service — Custom RAG with FastAPI, CrewAI, & LangGraph

This repository contains a modular, production-ready **Agentic RAG Service** built from scratch in Python. It progresses from basic vector retrieval to a robust web API, stateful routing, and multi-agent validation.

> **Zero Orchestration Bloat**: The core RAG retrieval, custom recursive chunking, and database operations are written in pure Python without using LangChain or LlamaIndex wrappers, showing exactly what happens under the hood. 

---

## 🚀 Key Features

*   **FastAPI REST Web Service**: Endpoints to list documents, upload/ingest PDF documents dynamically, and query context.
*   **CrewAI Multi-Agent Team**: A collaborative group of agents fact-checking QA outputs:
    *   **Document Retriever**: Uses a custom RAG database tool to gather relevant text.
    *   **Answer Synthesizer**: Drafts a concise answer grounded *only* in retrieved facts.
    *   **Answer Verifier**: A strict editor that cross-references assertions with sources and edits out hallucinations.
*   **LangGraph Routing State Machine**: A state graph routing logic:
    *   **Retrieve node** searches for context.
    *   **Conditional routing** checks if high-similarity chunks were found.
    *   **Answer node** fires if context is present, else routes directly to `END` to prevent hallucinations.
*   **Neon Database + pgvector**: Vector distance similarity matches are offloaded directly to Neon serverless PostgreSQL via SQL operators.
*   **Render-Ready Deployment**: Includes `render.yaml` and `requirements.txt` configs for instant hosting as a live Render Web Service.

---

## ⚙️ Directory Structure

```
test_rag/
│
├── api.py                    ← FastAPI Web Application Entry Point
├── config/
│   ├── __init__.py           ← Settings / configuration loader
│   └── openrouter_settings.py← OpenRouter LLM and embed models configuration
│
├── routes/
│   ├── health.py             ← Health status check endpoint
│   ├── documents.py          ← File uploading, listing, and deletion endpoints
│   └── chat.py               ← Querying and generating answers via LLM
│
├── services/
│   ├── __init__.py           
│   ├── agent_completion.py   ← OpenRouter API completion caller
│   ├── chat_service.py       ← Standard context search & generate service
│   └── ingest_service.py     ← PDF loading, chunking, embedding & vector database insertion
│
├── core/
│   ├── __init__.py           
│   ├── chunking.py           ← Recursive paragraph/sentence/word chunker
│   ├── document_loader.py    ← pypdf loader with page offset mapping
│   ├── embeddings.py         ← OpenRouter embed client
│   ├── models.py             ← Return structures (RetrievalResult, etc.)
│   ├── rag_engine.py         ← Prompt builders & contexts compiler
│   └── vector_store.py       ← Neon PostgreSQL connection and pgvector search queries
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
│   └── rag_graph.py          ← LangGraph StateGraph routing nodes (retrieve, answer, router)
│
├── scripts/
│   └── crew_runner.py        ← Script to kickoff the CrewAI RAG agent
│
├── docs/
│   └── project_flow.md       ← Full Mermaid diagrams
│
├── requirements.txt          ← System dependencies (for local & Render)
├── render.yaml               ← Deployment config for render.com Web Service
└── .gitignore                ← Files to exclude from git commits
```

---

## 📸 Architectural Visuals

Detailed diagrams showing how data flows through the application:

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
    J --> J1["INSERT INTO document_chunks\n(DOCUMENT_ID, chunk_index, chunk_text, embedding, page_number)"]
    
    J1 --> K["Returns IngestResult\n(DOCUMENT_ID, file_name, page_count, chunk_count)"]

    style A fill:#2a9d8f,color:#fff
    style B fill:#264653,color:#fff
    style K fill:#1b4332,color:#fff
    style H2 fill:#e76f51,color:#fff
    style I1 fill:#264653,color:#fff
    style J1 fill:#264653,color:#fff
```

### 2. Chatting with a PDF (API Query Flow)
```mermaid
flowchart TD
    A["POST /chat/query"] --> B["routes/chat.py (query_pdf)"]
    B --> C["ChatService.ask_pdf()"]
    
    C --> D["OpenRouterEmbeddingClient.embed_query()"]
    D --> D1["OpenRouter API → query vector"]
    
    D1 --> E["NeonVectorStore.similarity_search()"]
    E --> E1["pgvector cosine distance query\nFILTERED by DOCUMENT_ID\nLIMIT SEARCH_K=8"]
    E1 --> E2["list of RetrievalResult\n(score, chunk_id, chunk_text, page_number)"]
    
    E2 --> F{"Any score >= MIN_SCORE 0.3?"}
    F -- No --> F1["Return: not enough context"]
    F -- Yes --> G["Take top ANSWER_K=3 chunks\nwithin MAX_CONTEXT_CHARS=3000"]
    
    G --> H["RAGEngine.build_context()\nFormat: [Page X, Chunk Y]\nchunk_text..."]
    
    H --> I["RAGEngine.generate_answer()"]
    I --> I1["agent_complete() → OpenRouter LLM API"]
    I1 --> I2["LLM answers from context only"]
    
    I2 --> J["Returns ChatResult\n(answer, sources)"]
    J --> K["routes/chat.py returns QueryResponse JSON"]

    style A fill:#2a9d8f,color:#fff
    style B fill:#264653,color:#fff
    style F1 fill:#9b2226,color:#fff
    style D1 fill:#e76f51,color:#fff
    style E1 fill:#264653,color:#fff
    style I1 fill:#e76f51,color:#fff
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
        R_CHAT["routes/chat.py (query_pdf)"]
    end

    subgraph DI["Dependency Injection"]
        D1["dependencies.py"]
    end

    subgraph Services["Service Layer"]
        IS["IngestService.ingest_pdf()"]
        CS["ChatService.ask_pdf()"]
    end

    C1 --> R_DOC
    C2 --> R_CHAT

    R_DOC -.->|Depends| D1
    R_CHAT -.->|Depends| D1
    
    D1 -->|Injects| IS
    D1 -->|Injects| CS

    IS -->|Returns IngestResult| R_DOC
    CS -->|Returns ChatResult| R_CHAT

    R_DOC -->|Returns UploadResponse| C1
    R_CHAT -->|Returns QueryResponse| C2

    style C1 fill:#2a9d8f,color:#fff
    style C2 fill:#2a9d8f,color:#fff
    style R_DOC fill:#264653,color:#fff
    style R_CHAT fill:#264653,color:#fff
    style IS fill:#e76f51,color:#fff
    style CS fill:#e76f51,color:#fff
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
    TL -->|Queries| CS["ChatService().ask_pdf()"]
    
    T1 -->|Outputs Chunks| T2
    AG2 -->|Executes| T2
    
    T2 -->|Outputs Draft Answer| T3
    AG3 -->|Executes| T3
    T3 -->|Context checks| T1
    
    T3 -->|Returns Grounded Answer| C["Final Output Answer"]

    style A fill:#2d6a4f,color:#fff
    style CS fill:#e76f51,color:#fff
    style C fill:#1b4332,color:#fff
    style TL fill:#264653,color:#fff
```

### 5. LangGraph Self-Correcting Flow (Level 9)
```mermaid
flowchart TD
    A["graph/rag_graph.py (invoke)"] --> START["START"]
    
    START --> RET["retrieve_node"]
    
    RET --> GRADE["grade_documents_node\n(LLM grades context)"]
    GRADE --> COND1{"route_after_grading"}
    
    COND1 -->|has_context| SEL["select_context_node"]
    COND1 -->|retry_count < max| REW["rewrite_query_node\n(Expands search)"]
    COND1 -->|no retries left| NO_CTX["no_context_node"]
    
    REW --> RET
    
    SEL --> COND2{"route_after_select"}
    COND2 -->|has_context| ANS["answer_node\n(LLM drafts answer)"]
    COND2 -->|no_context| NO_CTX
    
    ANS --> HALLUC["check_hallucination_node\n(LLM verifies factuality)"]
    
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
| **PDF Parsing** | `pypdf` |
| **Database Driver** | `psycopg` (v3) |
| **Hosting Platform** | Render |

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/<your-username>/<your-new-repo-name>
cd <your-new-repo-name>
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

### Option A: Start the FastAPI Server
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` in your browser to test endpoints interactively using Swagger UI.

### Option B: Run the CrewAI Multi-Agent RAG Runner
```bash
python scripts/crew_runner.py
```

### Option C: Run the LangGraph Stateful Pipeline
```bash
python graph/rag_graph.py
```

---

## 🔌 API Reference

You can interact with the RAG backend in two ways: using the visual interactive UI, or programmatically via code.

### Method 1: Interactive UI (Swagger)
FastAPI automatically generates a visual dashboard where you can click **"Try it out"** to upload files and send queries directly from your browser.
👉 **Go to:** `http://127.0.0.1:8000/docs` (or your live Render URL `/docs`)

### Method 2: Programmatic API (Usage Example)
You can call the endpoints from any frontend (React, Flutter) or script using standard HTTP requests. 

**Usage example (Python `requests`):**
```python
import requests

BASE_URL = "https://your-render-url.com" # or http://127.0.0.1:8000

# 1. Upload a PDF
with open("sample.pdf", "rb") as f:
    upload_res = requests.post(f"{BASE_URL}/documents/upload", files={"file": f})
    
DOCUMENT_ID = upload_res.json()["DOCUMENT_ID"]
print(f"Uploaded successfully. Document ID: {DOCUMENT_ID}")

# 2. Ask a question about the PDF
payload = {
    "DOCUMENT_ID": DOCUMENT_ID,
    "question": "What is the main conclusion of this document?"
}
chat_res = requests.post(f"{BASE_URL}/chat/query", json=payload)

print("\n--- Answer ---")
print(chat_res.json()["answer"])
print(f"Latency: {chat_res.json()['process_time_ms']} ms")
```

### Available Endpoints
* **`POST /documents/upload`** — Uploads and vectorizes a `.pdf` file.
* **`GET /documents`** — Lists all indexed documents.
* **`DELETE /documents/{DOCUMENT_ID}`** — Removes a document and its vectors.
* **`POST /chat/query`** — Performs similarity search and generates an LLM answer.
* **`GET /health`** — Pinger endpoint to check server status.

---

## ☁️ Deploying on Render

This repository is pre-configured for deployment as a Render **Web Service** using the included `render.yaml` configuration.

### Deploy Steps:
1. Push your code to your new GitHub repository.
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
