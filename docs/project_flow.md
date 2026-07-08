# RAG Project — Code Flow Visual

Detailed diagrams showing how data flows through the application:

## Flow 1: Ingesting a PDF (API Flow)
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

## Flow 2: Chatting with a PDF (API Query Flow via LangGraph)
```mermaid
flowchart TD
    A["POST /chat/query"] --> B["routes/chat.py (query_pdf_graph)"]
    B --> C["GraphService.ask_pdf_with_graph()"]
    
    C --> D["rag_graph.invoke()"]
    D --> E["Retrieve Node → Grade Node → Answer Node"]
    E --> J["Returns ChatResult\n(answer, sources)"]
    J --> K["routes/chat.py returns QueryResponse JSON"]

    style A fill:#2a9d8f,color:#fff
    style B fill:#264653,color:#fff
    style D fill:#e76f51,color:#fff
    style E fill:#264653,color:#fff
    style J fill:#1b4332,color:#fff
```

## Flow 3: Web API Routing & Services
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

## Flow 5: LangGraph Self-Correcting Flow (Level 9)
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

## 🤖 LLM Completion Services

The application implements custom helper functions in [services/agent_completion.py](file:///home/sakil/Documents/LLM%20Learning/test_rag/services/agent_completion.py) to manage LLM interactions through OpenRouter:

*   **`agent_complete()`**: The primary completion function used for answering user queries.
    *   **Default Model**: `minimax/minimax-m2.5:free` (defined via `RouterModel.MINIMAX25.value`).
    *   **Usage**: Handles prompt synthesis and drafting responses from the context.
*   **`grading_complete()`**: A specialized completion function designed for grading and evaluation tasks.
    *   **Default Model**: `google/gemma-4-31b-it:free` (defined via `RouterGradingModel.GEMMA4.value`).
    *   **Usage**: Leveraged by the LangGraph pipeline in:
        *   `grade_documents_node` to evaluate if retrieved chunks are relevant to the user query.
        *   `check_hallucination_node` to verify that the generated answer is strictly grounded in the source context.

---
