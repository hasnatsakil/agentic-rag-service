# RAG Project — Code Flow Visual

## Flow 1: Ingesting a PDF

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

---

## Flow 2: Chatting with a PDF

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

---

## File Dependency Map

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
