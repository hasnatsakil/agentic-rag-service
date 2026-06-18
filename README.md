# 🔍 Pure Python RAG — Retrieval-Augmented Generation from Scratch

This repository tracks my LLM engineering journey. **We are now on Week 3!**

After building an in-memory RAG pipeline in Week 2, I have completely overhauled the architecture to use **Persistent Vector Storage (PostgreSQL + pgvector)**, **PDF Ingestion**, and exposed the entire pipeline via a **FastAPI REST API**.

> Every piece of this system is written in pure Python without using LangChain or LlamaIndex to understand exactly what happens under the hood.

---

## ✅ New Features in Week 3

- **FastAPI REST API Layer**: Exposed the RAG pipeline via clean REST endpoints for document upload, deletion, listing, and querying.
- **Pydantic Validation Schemas**: Added strong request/response schemas to validate inputs like similarity score threshold (`min_score`), retrieval limits (`search_k`, `answer_k`), and maximum character length.
- **Dependency Injection**: Leveraged FastAPI dependencies to cleanly inject core database (`NeonVectorStore`) and query/ingest services.
- **Persistent Vector Storage**: Migrated from in-memory arrays to a serverless Neon PostgreSQL database.
- **pgvector Integration**: Offloaded Cosine Similarity search directly into SQL using the `<=>` operator.
- **PDF Ingestion**: Added `pypdf` to load and extract text from PDFs, tracking exact page numbers for accurate citations.
- **Modular CLI Tools**: Retained command-line options for ingesting and chatting if you prefer working directly in the terminal.

---

## ⚙️ Core Architecture

```
test_rag/
│
├── CLI Entry Points
│   ├── ingest_pdf.py         ← Script to chunk and store PDFs
│   ├── pdf_chat_cli.py       ← Interactive RAG chat script
│   ├── list_documents.py     ← View stored documents
│   └── delete_document.py    ← Remove a document from the DB
│
├── Services & Core Logic
│   ├── chat_service.py       ← Orchestrates retrieval and LLM generation
│   ├── ingest_service.py     ← Orchestrates document loading, chunking, and storage
│   ├── rag_engine.py         ← RAG formatting and LLM completion logic
│   ├── chunking.py           ← Recursive paragraph/sentence/word chunking
│   ├── document_loader.py    ← PDF parsing and page tracking
│   ├── embeddings.py         ← OpenRouter embedding client
│   └── vector_store.py       ← NeonDB / psycopg integration with pgvector
│
└── Configuration
    ├── models.py             ← Data structures (RetrievalResult, etc.)
    └── openrouter_settings.py← API settings and LLM configuration
```

---

## 📸 Visuals & Execution Flows

### Flow 1: Ingesting a PDF

```mermaid
flowchart TD
    A["python ingest_pdf.py sample.pdf"] --> B["IngestService.ingest_pdf()"]
    
    B --> C["DocumentLoader.load_pdf_pages()"]
    C --> D["list of {page_number, text}"]
    
    D --> E["RecursiveChunk.chunk() per page"]
    E --> E1{"Paragraph fits?"}
    E1 -- Yes --> E2["Keep as chunk"]
    E1 -- No --> E3{"Sentence fits?"}
    E3 -- Yes --> E4["Keep as chunk"]
    E3 -- No --> E5["chunk_by_words(max_words=40, overlap=5)"]
    
    E2 & E4 & E5 --> F["all_chunks + page_numbers"]
    
    F --> G["RAGEngine.embed_chunks()"]
    G --> G1["OpenRouterEmbeddingClient.embed_documents()"]
    G1 --> G2["OpenRouter API → embedding vectors"]
    
    G2 --> H["NeonVectorStore.create_document()"]
    H --> H1["INSERT INTO documents → document_id"]
    
    H1 --> I["NeonVectorStore.insert_chunks()"]
    I --> I1["INSERT INTO document_chunks\n(document_id, chunk_index, chunk_text, embedding, page_number)"]
    
    I1 --> J["Returns IngestResult\n(document_id, file_name, page_count, chunk_count)"]

    style A fill:#2d6a4f,color:#fff
    style J fill:#1b4332,color:#fff
    style G2 fill:#e76f51,color:#fff
    style H1 fill:#264653,color:#fff
    style I1 fill:#264653,color:#fff
```

### Flow 2: Chatting with a PDF

```mermaid
flowchart TD
    A["python pdf_chat_cli.py"] --> B["NeonVectorStore.list_documents()"]
    B --> C["User picks document_id"]
    C --> D["User types question"]
    
    D --> E["ChatService.ask_pdf()"]
    
    E --> F["OpenRouterEmbeddingClient.embed_query()"]
    F --> F1["OpenRouter API → query vector"]
    
    F1 --> G["NeonVectorStore.similarity_search()"]
    G --> G1["pgvector cosine distance query\nFILTERED by document_id\nLIMIT search_k=8"]
    G1 --> G2["list of RetrievalResult\n(score, chunk_id, chunk_text, page_number)"]
    
    G2 --> H["print_retrieval_debug()\n✓ PASS / ✗ FAIL per chunk"]
    
    H --> I{"Any score >= min_score 0.3?"}
    I -- No --> I1["Return: not enough context"]
    I -- Yes --> J["Take top answer_k=3 chunks\nwithin max_context_chars=3000"]
    
    J --> K["RAGEngine.build_context()\nFormat: [Page X, Chunk Y]\nchunk_text..."]
    
    K --> L["RAGEngine.generate_answer()"]
    L --> L1["agent_complete() → OpenRouter LLM API"]
    L1 --> L2["LLM answers from context only"]
    
    L2 --> M["Returns ChatResult\n(answer, sources)"]
    M --> N["Print answer + source labels"]
    N --> D

    style A fill:#2d6a4f,color:#fff
    style I1 fill:#9b2226,color:#fff
    style F1 fill:#e76f51,color:#fff
    style G1 fill:#264653,color:#fff
    style L1 fill:#e76f51,color:#fff
    style M fill:#1b4332,color:#fff
```

### File Dependency Map

```mermaid
flowchart LR
    subgraph CLI["CLI Entry Points"]
        IP["ingest_pdf.py"]
        PC["pdf_chat_cli.py"]
    end

    subgraph Services["Service Layer"]
        IS["ingest_service.py"]
        CS["chat_service.py"]
    end

    subgraph Core["Core Logic"]
        RE["rag_engine.py"]
        CH["chunking.py"]
        DL["document_loader.py"]
        EM["embeddings.py"]
        VS["vector_store.py"]
    end

    subgraph Config["Config & Models"]
        OR["openrouter_settings.py"]
        MD["models.py"]
    end

    subgraph External["External Services"]
        NE["Neon PostgreSQL\n+ pgvector"]
        OA["OpenRouter API\n(embeddings + LLM)"]
    end

    IP --> IS
    PC --> CS
    PC --> VS

    IS --> DL
    IS --> CH
    IS --> RE
    IS --> VS
    IS --> MD

    CS --> EM
    CS --> RE
    CS --> VS
    CS --> MD

    RE --> EM
    RE --> AC["agent_completion.py"]
    RE --> MD

    EM --> OR
    AC --> OR
    VS --> MD

    OR --> OA
    VS --> NE

    style IP fill:#2d6a4f,color:#fff
    style PC fill:#2d6a4f,color:#fff
    style NE fill:#264653,color:#fff
    style OA fill:#e76f51,color:#fff
```

---

## 🏗️ Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI (with Pydantic schemas) |
| Web Server | Uvicorn |
| LLM API | OpenRouter (`openrouter.ai`) |
| Embeddings | `openai/text-embedding-3-small` via OpenRouter |
| Vector DB | Neon Serverless PostgreSQL |
| Vector Search | `pgvector` (Cosine Similarity) |
| Database Driver | `psycopg` (v3) |
| PDF Parsing | `pypdf` |
| Chunking | Pure Python Recursive Chunking |

---

## 🚀 Getting Started

### 1. Clone and set up environment

```bash
git clone https://github.com/hasnatsakil/pure-python-rag
cd pure-python-rag
python -m venv venv
source venv/bin/activate
pip install psycopg[binary] pypdf pydantic structlog python-dotenv openai fastapi uvicorn python-multipart
```

### 2. Configure Environment

Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DATABASE_URL=postgresql://user:password@ep-cool-db.region.aws.neon.tech/dbname
```

### 3. Usage

#### Option A: Running the FastAPI Web Server (New!)

**Start the server:**
```bash
uvicorn api:app --reload
```

**Explore the API Docs:**
Open `http://127.0.0.1:8000/docs` in your browser to access the interactive Swagger UI.

**Example Endpoints:**
* **Upload a PDF:**
  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/documents/upload' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@sample.pdf;type=application/pdf'
  ```
* **List Documents:**
  ```bash
  curl -X 'GET' 'http://127.0.0.1:8000/documents'
  ```
* **Ask a Question:**
  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/chat/query' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "document_id": 1,
    "question": "What is the main topic of this paper?",
    "search_k": 15,
    "answer_k": 5,
    "min_score": 0.3,
    "max_context_chars": 5000
  }'
  ```

#### Option B: Terminal Command Line Interfaces

**Ingest a PDF:**
```bash
python ingest_pdf.py sample.pdf
```

**Chat with your PDF:**
```bash
python pdf_chat_cli.py
```

---

## 🔜 Next Phase

- Expanding vector search with hybrid search (BM25 + pgvector).
- Building a modern frontend UI (Next.js / React) to chat with uploaded PDFs visually.

---

## 📜 License

MIT

