# RAG Project — Code Flow Visual

## Flow 1: Ingesting a PDF (API)

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
    I --> I1["INSERT INTO documents → document_id"]
    
    I1 --> J["NeonVectorStore.insert_chunks()"]
    J --> J1["INSERT INTO document_chunks\n(document_id, chunk_index, chunk_text, embedding, page_number)"]
    
    J1 --> K["Returns IngestResult\n(document_id, file_name, page_count, chunk_count)"]

    style A fill:#2a9d8f,color:#fff
    style B fill:#264653,color:#fff
    style K fill:#1b4332,color:#fff
    style H2 fill:#e76f51,color:#fff
    style I1 fill:#264653,color:#fff
    style J1 fill:#264653,color:#fff
```

---

## Flow 2: Chatting with a PDF (API Query)

```mermaid
flowchart TD
    A["POST /chat/query"] --> B["routes/chat.py (query_pdf)"]
    B --> C["ChatService.ask_pdf()"]
    
    C --> D["OpenRouterEmbeddingClient.embed_query()"]
    D --> D1["OpenRouter API → query vector"]
    
    D1 --> E["NeonVectorStore.similarity_search()"]
    E --> E1["pgvector cosine distance query\nFILTERED by document_id\nLIMIT search_k=8"]
    E1 --> E2["list of RetrievalResult\n(score, chunk_id, chunk_text, page_number)"]
    
    E2 --> F{"Any score >= min_score 0.3?"}
    F -- No --> F1["Return: not enough context"]
    F -- Yes --> G["Take top answer_k=3 chunks\nwithin max_context_chars=3000"]
    
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


---

## Flow 3: FastAPI Web API Endpoints

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

---

## Flow 4: CrewAI Collaborative Team Flow

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

---

## Flow 5: LangGraph Routing State Machine Flow

```mermaid
flowchart TD
    A["graph/rag_graph.py (invoke)"] --> START["START"]
    
    START --> RET["retrieve_node"]
    
    RET -->|Calls OpenRouter Embeddings| EM["core/embeddings.py"]
    RET -->|Queries similarity search| VS["core/vector_store.py"]
    
    VS --> RET_OUT["Filtered context chunks (score >= 0.3)"]
    
    RET_OUT --> COND{"check_context router"}
    
    COND -->|has_context == True| ANS["answer_node"]
    COND -->|has_context == False| END1["END (Return: No relevant info)"]
    
    ANS -->|Calls generation| RE["core/rag_engine.py"]
    RE --> END2["END (Return: Final Answer)"]

    style A fill:#2d6a4f,color:#fff
    style COND fill:#e76f51,color:#fff
    style END1 fill:#9b2226,color:#fff
    style END2 fill:#1b4332,color:#fff
```

---

## File Dependency Map

```mermaid
flowchart TD
    subgraph Entry["Entry Points"]
        API["api.py (FastAPI App)"]
        CR["scripts/crew_runner.py (CLI)"]
        GR["graph/rag_graph.py (CLI)"]
    end

    subgraph API_Routes["API Routes"]
        RH["routes/health.py"]
        RD["routes/documents.py"]
        RC["routes/chat.py"]
    end

    subgraph Dependency["Dependency Injection"]
        DEP["dependencies.py"]
    end

    subgraph Agentic["Agentic Layers"]
        CRW["crew/crews/rag_crew.py (CrewAI)"]
        TL["crew/tools/rag_tool.py (RAG Tool)"]
        LGP["graph/rag_graph.py (LangGraph)"]
    end

    subgraph Services["Service Layer"]
        IS["services/ingest_service.py"]
        CS["services/chat_service.py"]
    end

    subgraph Core["Core Logic"]
        RE["core/rag_engine.py"]
        CH["core/chunking.py"]
        DL["core/document_loader.py"]
        EM["core/embeddings.py"]
        VS["core/vector_store.py"]
    end

    subgraph Config_Models["Config, Models & Schemas"]
        CONF["config/__init__.py"]
        SCH["schemas.py"]
        OR["config/openrouter_settings.py"]
        MD["core/models.py"]
    end

    subgraph External["External Infrastructure"]
        NE["Neon PostgreSQL + pgvector"]
        OA["OpenRouter API"]
    end

    %% Routing Flow
    API --> RH
    API --> RD
    API --> RC

    %% Dependency Connections
    RD -.-> DEP
    RC -.-> DEP
    DEP --> IS
    DEP --> CS
    DEP --> VS

    %% CLI and Runners Connections
    CR --> CRW
    CRW --> TL
    TL --> CS
    GR --> LGP
    LGP --> EM
    LGP --> VS
    LGP --> RE

    %% Service Connections
    IS --> DL
    IS --> CH
    IS --> RE
    IS --> VS
    
    CS --> EM
    CS --> RE
    CS --> VS

    %% Core Connections
    RE --> EM
    RE --> AC["services/agent_completion.py"]
    EM --> OR
    AC --> OR
    VS --> NE

    %% Configuration & Schema usage
    RD & RC --> SCH
    SCH --> CONF
    VS & IS & CS & RE --> MD
    OR & AC & EM --> OA

    style API fill:#e76f51,color:#fff
    style CR fill:#2d6a4f,color:#fff
    style GR fill:#2d6a4f,color:#fff
    style NE fill:#264653,color:#fff
    style OA fill:#264653,color:#fff
```


