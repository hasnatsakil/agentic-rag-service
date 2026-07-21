# RAG Project — Code Flow Diagrams

Detailed Mermaid diagrams showing how data flows through every layer of the application.

---

## Flow 1 — PDF Ingestion (`POST /documents/upload`)

```mermaid
flowchart TD
    A["POST /documents/upload"] --> B["routes/documents.py\nupload_file()"]
    B --> C["IngestService.ingest_pdf()"]

    C --> D["DocumentLoader.load_pdf_pages()"]
    D --> E["list of {page_number, text} per page"]

    E --> F["RecursiveChunk.chunk() per page"]
    F --> F1{"Paragraph fits ≤ max_words?"}
    F1 -- Yes --> F2["Keep paragraph as chunk"]
    F1 -- No --> F3{"Sentence fits?"}
    F3 -- Yes --> F4["Keep sentence as chunk"]
    F3 -- No --> F5["chunk_by_words()\nsliding window with overlap"]

    F2 & F4 & F5 --> G["all_chunks + page_numbers lists"]

    G --> H["RAGEngine.embed_chunks()"]
    H --> H1["OpenRouterEmbeddingClient\n(auto model-fallback)"]
    H1 --> H2["OpenRouter API → embedding vectors [1536d]"]

    G --> SUM["agent_complete()\nGenerate one-sentence document summary"]

    H2 --> I["NeonVectorStore.create_document(file_name, summary)"]
    I --> I1["INSERT INTO documents → document_id"]

    I1 --> J["NeonVectorStore.insert_chunks()"]
    J --> J1["INSERT INTO document_chunks\n(document_id, chunk_index, chunk_text, embedding, page_number)"]

    J1 --> K["Returns IngestResult\n(document_id, file_name, page_count, chunk_count)"]

    style A fill:#2a9d8f,color:#fff
    style K fill:#1b4332,color:#fff
    style H2 fill:#e76f51,color:#fff
    style SUM fill:#e76f51,color:#fff
    style I1 fill:#264653,color:#fff
    style J1 fill:#264653,color:#fff
```

---

## Flow 2 — Agentic Chat Query (`POST /chat/query`)

```mermaid
flowchart TD
    A["POST /chat/query\n{session_id, question, ...}"] --> B["routes/chat.py\nquery_pdf_graph()"]

    B --> MEM1["ChatHistoryStore.get_last_20_message(session_id)"]
    B --> MEM2["ChatHistoryStore.get_summary(session_id)"]

    MEM1 & MEM2 --> C["GraphService.ask_pdf_with_graph()"]
    C --> D["rag_graph.invoke(RAGState)"]

    D --> AGENT["agent_node\nLLM + tool schema\nReceives: history, summary, doc catalog"]

    AGENT -- "Direct answer\n(no tool call needed)" --> HALL["check_hallucination_node\nVerifier LLM: is answer grounded?"]
    AGENT -- "Tool call:\nsearch_pdf_database(query, document_id)" --> TOOL["execute_tool_node\nhybrid_search (vector + keyword)\n→ RRF fusion\n→ Pass 1 lexical reranking"]

    TOOL --> OPT{"use_llm_rerank = true?"}
    OPT -- Yes --> LLM_R["llm_rerank_node\nPass 2: LLM judge\nreorders chunks best-first"]
    OPT -- No --> GRADE
    LLM_R --> GRADE["grade_documents_node\nJudge LLM: yes / no per chunk"]

    GRADE -- "Relevant chunks found" --> SEL["select_context_node\nTrim to ANSWER_K / MAX_CONTEXT_CHARS"]
    GRADE -- "Retry budget left" --> REW["rewrite_query_node\nLLM rewrites query as keywords"]
    GRADE -- "Retries exhausted" --> NO["no_context_node\nReturn fallback message"]

    REW --> TOOL
    SEL --> AGENT

    HALL --> END_OK["END\nChatResult(answer, sources, debug)"]
    NO --> END_OK

    END_OK --> BG["BackgroundTask\nsave_and_compact_workflow()"]
    BG --> BG1["ChatHistoryStore.save_message() × 2\n(user + assistant turn)"]
    BG --> BG2{"history >= 10 messages?"}
    BG2 -- Yes --> BG3["LLM: merge turn into\nrunning summary (≤3 sentences)"]
    BG3 --> BG4["ChatHistoryStore.update_summary()"]
    BG2 -- No --> BG5["Skip summary update"]

    END_OK --> RESP["QueryResponse JSON\n{answer, sources, debug, process_time_ms}"]

    style A fill:#2a9d8f,color:#fff
    style END_OK fill:#1b4332,color:#fff
    style HALL fill:#e76f51,color:#fff
    style GRADE fill:#e76f51,color:#fff
    style LLM_R fill:#e76f51,color:#fff
    style REW fill:#e76f51,color:#fff
    style BG3 fill:#e76f51,color:#fff
```

---

## Flow 3 — Hybrid Search & Two-Pass Re-ranking Detail

```mermaid
flowchart TD
    Q["search_question\n(from agent tool call)"] --> EMB["OpenRouterEmbeddingClient.embed_query()"]

    EMB --> VS["NeonVectorStore.similarity_search()\npgvector cosine distance <=>"]
    Q --> KS["NeonVectorStore.keyword_search()\nts_rank + websearch_to_tsquery"]

    VS & KS --> RRF["Reciprocal Rank Fusion\nRRF(d) = Σ 1 / (60 + rank)\nProduces unified ranked list"]

    RRF --> P1["RAGEngine.rerank_results()\nPass 1 — keyword overlap boost\n(lightweight, always applied)"]

    P1 --> FILTER["Filter: score >= MIN_SCORE"]

    FILTER --> OPT{"use_llm_rerank?"}
    OPT -- Yes --> P2["llm_rerank_node\nPass 2 — LLM-as-a-Judge\nreorders candidates best-first"]
    OPT -- No --> GRADE_IN["grade_documents_node"]
    P2 --> GRADE_IN

    GRADE_IN --> SEL["select_context_node\nApply ANSWER_K + MAX_CONTEXT_CHARS budget"]
    SEL --> CTX["Final context string\npassed back to agent_node"]

    style RRF fill:#264653,color:#fff
    style P1 fill:#457b9d,color:#fff
    style P2 fill:#e76f51,color:#fff
    style CTX fill:#1b4332,color:#fff
```

---

## Flow 4 — Multi-Turn Memory & Session Compaction

```mermaid
flowchart TD
    subgraph "Per Request — Memory Injection"
        R["POST /chat/query\n(session_id, question)"]
        R --> HS["ChatHistoryStore.get_last_20_message(session_id)\nFetches recent turns in chrono order"]
        R --> SS["ChatHistoryStore.get_summary(session_id)\nFetches running summary of older turns"]
        HS & SS --> GS["GraphService → RAGState\nhistory: list[dict]\nsummary: str"]
        GS --> AG["agent_node system prompt receives:\n• Recent chat history (last 20)\n• Summary of older history\n• Available document catalog\n• User question"]
    end

    subgraph "After Response — Background Compaction"
        BG["save_and_compact_workflow()\nFastAPI BackgroundTask"]
        BG --> SM["save_message(user) + save_message(assistant)"]
        SM --> CHK{"Total messages >= 10?"}
        CHK -- Yes --> LLM_S["agent_complete(is_grading=True)\nIntegrate latest turn into\nexisting summary (≤3 sentences)"]
        LLM_S --> US["ChatHistoryStore.update_summary()\nUPSERT into chat_summaries table"]
        CHK -- No --> SKIP["Skip — summary not yet needed"]
    end

    style AG fill:#264653,color:#fff
    style LLM_S fill:#e76f51,color:#fff
    style US fill:#1b4332,color:#fff
```

---

## Flow 5 — API Routing & Service Layer

```mermaid
flowchart TD
    subgraph Client["HTTP Clients (Postman / curl / Frontend)"]
        C1["POST /documents/upload"]
        C2["GET /documents"]
        C3["DELETE /documents/{id}"]
        C4["POST /chat/query"]
        C5["GET /chat/sessions"]
        C6["DELETE /chat/sessions/{session_id}"]
    end

    subgraph FastAPI["FastAPI Layer (api.py)"]
        R_DOC["routes/documents.py"]
        R_CHAT["routes/chat.py"]
        R_HEALTH["routes/health.py"]
    end

    subgraph DI["Dependency Injection (dependencies.py)"]
        D_VS["get_vector_store() → NeonVectorStore"]
        D_IS["get_ingest_service() → IngestService"]
    end

    subgraph Services["Service / Core Layer"]
        IS["IngestService.ingest_pdf()"]
        GS["GraphService.ask_pdf_with_graph()"]
        HS["ChatHistoryStore"]
        CS["save_and_compact_workflow()"]
    end

    C1 --> R_DOC
    C2 --> R_DOC
    C3 --> R_DOC
    C4 --> R_CHAT
    C5 --> R_CHAT
    C6 --> R_CHAT

    R_DOC -. Depends .-> D_IS & D_VS
    R_CHAT -. Depends .-> D_VS

    D_IS --> IS
    R_CHAT --> GS
    R_CHAT --> HS
    GS --> CS

    style C1 fill:#2a9d8f,color:#fff
    style C4 fill:#2a9d8f,color:#fff
    style IS fill:#e76f51,color:#fff
    style GS fill:#e76f51,color:#fff
```

---

## Flow 6 — CrewAI Multi-Agent Pipeline

```mermaid
flowchart TD
    A["rag_crew.kickoff({question})"] --> B["Sequential Task Pipeline"]

    subgraph Crew["Sequential CrewAI Crew"]
        T1["Task 1 — Retrieve\nretriever_agent\n→ calls search_pdf tool"]
        T2["Task 2 — Synthesize\nanswer_agent\n→ drafts grounded answer from chunks"]
        T3["Task 3 — Verify\nverifier_agent\n→ fact-checks + corrects answer"]
    end

    B --> T1
    T1 --> TL["crew/tools/rag_tool.py\nsearch_pdf(query)"]
    TL --> GS["GraphService.ask_pdf_with_graph()\n(full LangGraph pipeline)"]
    GS --> TL
    TL --> T1

    T1 -- "chunks as context" --> T2
    T2 -- "draft answer" --> T3
    T3 --> OUT["Final verified answer\nprinted to stdout"]

    style A fill:#2d6a4f,color:#fff
    style GS fill:#e76f51,color:#fff
    style OUT fill:#1b4332,color:#fff
    style TL fill:#264653,color:#fff
```

---

## Why This Architecture?

Standard RAG pipelines blindly retrieve documents and hand them to the LLM, causing
hallucinations when the context is poor. This service addresses each failure mode explicitly:

| Problem | Solution |
|---|---|
| LLM answers from stale memory instead of searching | Native tool calling — LLM must call `search_pdf_database` and pick the correct document ID |
| Wrong document searched | Agent receives a full catalog with IDs and summaries; it chooses the ID autonomously |
| Poor context quality | `grade_documents_node` evaluates each chunk independently and discards irrelevant ones |
| Retrieval misses the right chunks | `rewrite_query_node` uses an LLM to rephrase the query as keywords before retrying |
| LLM answers go beyond source facts | `check_hallucination_node` cross-references the answer against retrieved chunks |
| Context window overflow over long sessions | `compaction_service` maintains a rolling ≤3-sentence summary in PostgreSQL |
