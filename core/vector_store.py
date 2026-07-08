"""
Neon PostgreSQL vector store for document chunk storage and retrieval.

This module provides :class:`NeonVectorStore`, which persists document chunks
and their embeddings in a Neon (serverless Postgres) database using the
``pgvector`` extension.  It exposes three complementary search strategies:

- **Vector search** — approximate nearest-neighbour using cosine distance
  (``<=>``), ideal for semantic similarity.
- **Keyword search** — full-text search using PostgreSQL ``ts_rank`` and
  ``websearch_to_tsquery``, ideal for exact term matching.
- **Hybrid search** — fuses both via Reciprocal Rank Fusion (RRF), combining
  the strengths of each approach.

Database schema expected:

.. code-block:: sql

    CREATE TABLE documents (
        id         SERIAL PRIMARY KEY,
        file_name  TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE document_chunks (
        id          SERIAL PRIMARY KEY,
        DOCUMENT_ID INTEGER REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        chunk_text  TEXT NOT NULL,
        embedding   VECTOR(1536),
        page_number INTEGER
    );

Environment variables:
    DATABASE_URL: Full Neon PostgreSQL connection string, loaded from
        ``.env`` via :mod:`dotenv`.
"""

import os

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv
from core.models import RetrievalResult
from pgvector.psycopg import register_vector
from config import settings

load_dotenv()


class NeonVectorStore:
    """PostgreSQL-backed vector store using Neon and pgvector.

    All methods are class-methods; no instantiation is required.  Each method
    opens and closes its own database connection via the context manager
    returned by :meth:`connect`.

    Class Attributes:
        DATABASE_URL (str | None): Neon connection string from the
            ``DATABASE_URL`` environment variable.
        TOP_K (int): Default number of results to return from search methods.
            Sourced from ``settings.SEARCH_K``.
        DOCUMENT_ID (int): Default document ID used when no explicit ID is
            provided. Sourced from ``settings.DOCUMENT_ID``.
    """

    DATABASE_URL = os.getenv("DATABASE_URL")
    TOP_K: int = settings.SEARCH_K
    DOCUMENT_ID: int = settings.DOCUMENT_ID

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def connect(cls) -> psycopg.Connection:
        """Open and return a new database connection with pgvector registered.

        Registers the ``pgvector`` type adapter on the connection so that
        Python lists of floats are automatically serialised to and from the
        Postgres ``vector`` type.

        Returns:
            An open :class:`psycopg.Connection` instance.

        Raises:
            ValueError: If ``DATABASE_URL`` is not set in the environment.
            psycopg.OperationalError: If the connection cannot be established.
        """
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is missing from .env")
        conn = psycopg.connect(cls.DATABASE_URL)
        register_vector(conn)
        return conn

    @classmethod
    def test_connection(cls) -> None:
        """Verify database connectivity and print the server version.

        Executes a lightweight ``SELECT version()`` query and prints the
        result to stdout.  Intended for use in health checks and local
        development smoke tests.

        Raises:
            ValueError: If ``DATABASE_URL`` is missing.
            psycopg.OperationalError: If the database is unreachable.
        """
        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]

        print("Connected to NeonDB:")
        print(version)

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def create_document(cls, file_name: str) -> int:
        """Insert a new document record and return its generated ID.

        Args:
            file_name: The original filename of the uploaded PDF.

        Returns:
            The auto-generated integer primary key (``id``) of the newly
            created row in the ``documents`` table.
        """
        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO documents (file_name)
                    VALUES (%s)
                    RETURNING id;
                    """,
                    (file_name,),
                )
                DOCUMENT_ID = cursor.fetchone()[0]

        return DOCUMENT_ID

    @classmethod
    def insert_chunks(
        cls,
        DOCUMENT_ID: int,
        chunks: list[str],
        embeddings: list[list[float]],
        page_numbers: list[int] | None = None,
    ) -> None:
        """Batch-insert text chunks and their embeddings into the vector store.

        Uses :meth:`psycopg.Cursor.executemany` for efficient bulk insertion.
        When ``page_numbers`` is ``None``, all chunks are recorded without
        page associations.

        Args:
            DOCUMENT_ID: The ID of the parent document (from
                :meth:`create_document`).
            chunks: List of text strings to store.
            embeddings: Embedding vectors corresponding to each chunk.
                Must be the same length as ``chunks``.
            page_numbers: Optional list of 1-based page numbers for each
                chunk.  Must be the same length as ``chunks`` when provided.
                Defaults to a list of ``None`` values.

        Raises:
            ValueError: If ``chunks``, ``embeddings``, and ``page_numbers``
                do not all have the same length.
        """
        if page_numbers is None:
            page_numbers = [None] * len(chunks)
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if len(chunks) != len(page_numbers):
            raise ValueError("chunks and page_numbers must have the same length")

        rows = []
        for chunk_index, chunk_text in enumerate(chunks):
            rows.append((
                DOCUMENT_ID,
                chunk_index,
                chunk_text,
                embeddings[chunk_index],
                page_numbers[chunk_index],
            ))

        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO document_chunks (
                        DOCUMENT_ID,
                        chunk_index,
                        chunk_text,
                        embedding,
                        page_number
                    )
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    rows,
                )

    # ------------------------------------------------------------------ #
    #  Search operations                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def similarity_search(
        cls,
        query_embedding: list[float],
        top_k: int = TOP_K,
        DOCUMENT_ID: int | None = None,
    ) -> list[RetrievalResult]:
        """Search for chunks by vector (semantic) similarity.

        Uses the pgvector cosine-distance operator (``<=>``) to find the
        ``top_k`` chunks closest to the query embedding.  The raw distance
        is converted to a similarity score as ``score = 1 - distance``.

        Args:
            query_embedding: The embedding vector of the search query,
                produced by :meth:`~core.embeddings.OpenRouterEmbeddingClient.embed_query`.
            top_k: Maximum number of results to return. Defaults to
                :attr:`TOP_K`.
            DOCUMENT_ID: When provided, restricts results to chunks
                belonging to this document.  When ``None``, searches across
                all documents.

        Returns:
            A list of :class:`~core.models.RetrievalResult` objects with
            ``retrieval_method="vector"``, sorted by descending similarity
            score.
        """
        with cls.connect() as connection:
            with connection.cursor() as cursor:
                if DOCUMENT_ID is None:
                    cursor.execute(
                        """
                        SELECT
                            embedding <=> %s::vector AS distance,
                            chunk_index,
                            chunk_text,
                            page_number
                        FROM document_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (query_embedding, query_embedding, top_k),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT
                            embedding <=> %s::vector AS distance,
                            chunk_index,
                            chunk_text,
                            page_number
                        FROM document_chunks
                        WHERE DOCUMENT_ID = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (query_embedding, DOCUMENT_ID, query_embedding, top_k),
                    )
                rows = cursor.fetchall()

        results: list[RetrievalResult] = []
        for distance, chunk_index, chunk_text, page_number in rows:
            # Convert cosine distance to similarity: closer = higher score.
            similarity_score = 1 - distance
            results.append(
                RetrievalResult(
                    score=similarity_score,
                    chunk_id=chunk_index,
                    chunk_text=chunk_text,
                    page_number=page_number,
                    retrieval_method="vector",
                )
            )

        return results

    @classmethod
    def keyword_search(
        cls,
        query: str,
        top_k: int = TOP_K,
        DOCUMENT_ID: int | None = None,
    ) -> list[RetrievalResult]:
        """Search for chunks by full-text keyword matching.

        Converts the query into a PostgreSQL ``websearch_to_tsquery``
        expression (terms joined with ``OR``) and ranks results using
        ``ts_rank``.  Best suited for exact term or phrase lookups.

        Args:
            query: The natural-language search query.
            top_k: Maximum number of results to return. Defaults to
                :attr:`TOP_K`.
            DOCUMENT_ID: When provided, restricts results to this document.
                When ``None``, searches across all documents.

        Returns:
            A list of :class:`~core.models.RetrievalResult` objects with
            ``retrieval_method="keyword"``, sorted by descending ``ts_rank``
            score.
        """
        # Join all query words with OR so any term can match.
        query = " OR ".join(query.split())

        with cls.connect() as connection:
            with connection.cursor() as cursor:
                if DOCUMENT_ID is None:
                    cursor.execute(
                        """
                        SELECT
                            ts_rank(
                                to_tsvector('english', chunk_text),
                                websearch_to_tsquery('english', %s)
                            ) AS keyword_score,
                            chunk_index,
                            chunk_text,
                            page_number
                        FROM document_chunks
                        WHERE to_tsvector('english', chunk_text) @@ websearch_to_tsquery('english', %s)
                        ORDER BY keyword_score DESC
                        LIMIT %s;
                        """,
                        (query, query, top_k),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT
                            ts_rank(
                                to_tsvector('english', chunk_text),
                                websearch_to_tsquery('english', %s)
                            ) AS keyword_score,
                            chunk_index,
                            chunk_text,
                            page_number
                        FROM document_chunks
                        WHERE DOCUMENT_ID = %s
                        AND to_tsvector('english', chunk_text) @@ websearch_to_tsquery('english', %s)
                        ORDER BY keyword_score DESC
                        LIMIT %s;
                        """,
                        (query, DOCUMENT_ID, query, top_k),
                    )
                rows = cursor.fetchall()

        results: list[RetrievalResult] = []
        for keyword_score, chunk_index, chunk_text, page_number in rows:
            results.append(
                RetrievalResult(
                    score=float(keyword_score),
                    chunk_id=chunk_index,
                    chunk_text=chunk_text,
                    page_number=page_number,
                    retrieval_method="keyword",
                )
            )

        return results

    @classmethod
    def hybrid_search(
        cls,
        query: str,
        query_embedding: list[float],
        top_k: int = TOP_K,
        DOCUMENT_ID: int | None = None,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Search using Reciprocal Rank Fusion over vector and keyword results.

        Runs :meth:`similarity_search` and :meth:`keyword_search` independently,
        then merges their ranked lists using the RRF formula:

        .. math::

            \\text{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k + r(d)}

        where ``k`` is :attr:`rrf_k` and ``r(d)`` is the rank of document
        ``d`` in result list ``r``.  Chunks that appear in both result sets
        receive a higher combined score.

        Args:
            query: The natural-language search query (used for keyword search).
            query_embedding: The embedded query vector (used for vector search).
            top_k: Number of final fused results to return. Defaults to
                :attr:`TOP_K`.
            DOCUMENT_ID: When provided, restricts both sub-searches to this
                document.
            vector_k: Number of candidates to retrieve from vector search.
                Defaults to ``top_k``.
            keyword_k: Number of candidates to retrieve from keyword search.
                Defaults to ``top_k``.
            rrf_k: Smoothing constant for the RRF formula. Higher values
                reduce the impact of rank differences. Defaults to ``60``.

        Returns:
            A list of :class:`~core.models.RetrievalResult` objects with
            ``retrieval_method="hybrid"``, sorted by descending RRF score,
            containing at most ``top_k`` entries.
        """
        if vector_k is None:
            vector_k = top_k
        if keyword_k is None:
            keyword_k = top_k

        vector_results = cls.similarity_search(
            query_embedding=query_embedding,
            top_k=vector_k,
            DOCUMENT_ID=DOCUMENT_ID,
        )
        keyword_results = cls.keyword_search(
            query=query,
            top_k=keyword_k,
            DOCUMENT_ID=DOCUMENT_ID,
        )

        # Accumulate RRF scores keyed by (page_number, chunk_id).
        merged: dict[tuple, dict] = {}

        def add_results(results: list[RetrievalResult], source: str) -> None:
            """Add ranked results from one retrieval strategy to the merged dict."""
            for rank, result in enumerate(results, start=1):
                key = (result.page_number, result.chunk_id)

                if key not in merged:
                    merged[key] = {"result": result, "score": 0.0, "sources": []}

                # RRF contribution from this strategy.
                merged[key]["score"] += 1 / (rrf_k + rank)
                merged[key]["sources"].append(source)

        add_results(vector_results, "vector")
        add_results(keyword_results, "keyword")

        hybrid_results: list[RetrievalResult] = []
        for item in merged.values():
            result = item["result"]
            result.score = item["score"]
            result.retrieval_method = "hybrid"
            hybrid_results.append(result)

        hybrid_results.sort(key=lambda r: r.score, reverse=True)
        return hybrid_results[:top_k]

    # ------------------------------------------------------------------ #
    #  Document management                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def list_documents(cls) -> list[dict[str, object]]:
        """Retrieve a list of all ingested documents from the database.

        Returns:
            A list of dicts, one per document, with the keys ``"id"``,
            ``"file_name"``, and ``"created_at"`` (ISO-format string).
            Results are ordered by creation time descending (newest first).
        """
        with cls.connect() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id,
                           file_name,
                           created_at::text
                    FROM documents
                    ORDER BY created_at DESC;
                    """
                )
                rows = cursor.fetchall()

        return rows

    @classmethod
    def delete_document(cls, DOCUMENT_ID: int) -> None:
        """Delete a document and all its associated chunks from the database.

        Because ``document_chunks`` references ``documents`` with
        ``ON DELETE CASCADE``, deleting the parent row automatically removes
        all child chunk rows.

        Args:
            DOCUMENT_ID: The primary key of the document to delete.
        """
        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s;
                    """,
                    (DOCUMENT_ID,),
                )
