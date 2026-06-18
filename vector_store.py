import os

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv
from models import RetrievalResult
from pgvector.psycopg import register_vector

load_dotenv()

class NeonVectorStore:
    DATABASE_URL = os.getenv("DATABASE_URL")

    @classmethod
    def connect(
        cls
        ) -> psycopg.Connection:

        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is missing from .env")
        conn = psycopg.connect(cls.DATABASE_URL)
        register_vector(conn)

        return conn
    
    @classmethod
    def test_connection(
        cls
        ) -> bool:

        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
        
        print("Connected to NeonDB:")
        print(version)

    @classmethod
    def create_document(
        cls,
        file_name: str
        ) -> int:
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
                document_id = cursor.fetchone()[0]

        return document_id
    
    @classmethod
    def insert_chunks(
        cls,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]],
        page_numbers: list[int] | None = None
        ) -> None:
        if page_numbers is None:
            page_numbers = [None] * len(chunks)
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if len(chunks) != len(page_numbers):
            raise ValueError("chunks and page_numbers must have the same length")
        rows = []
        for chunk_index, chunk_text in enumerate(chunks):
            rows.append(
                (
                    document_id, 
                    chunk_index, 
                    chunk_text, 
                    embeddings[chunk_index],
                    page_numbers[chunk_index],
                )
            )
        
        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                        """
                        INSERT INTO document_chunks (
                        document_id,
                        chunk_index, 
                        chunk_text, 
                        embedding,
                        page_number
                        )
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        rows
                    )
    @classmethod
    def similarity_search(
        cls,
        query_embedding: list[float],
        top_k: int = 3,
        document_id: int | None = None
        ) -> list[RetrievalResult]:

        with cls.connect() as connection:
            with connection.cursor() as cursor:
                if document_id is None:
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
                        (
                            query_embedding,
                            query_embedding,
                            top_k
                        ),
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
                        WHERE document_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (
                            query_embedding,
                            document_id,
                            query_embedding,
                            top_k
                        ),
                    )
                rows = cursor.fetchall()
        results = []

        for distance, chunk_index, chunk_text, page_number in rows:
            similarity_score = 1 - distance
            results.append(
                RetrievalResult(
                    score=similarity_score,
                    chunk_id=chunk_index,
                    chunk_text=chunk_text,
                    page_number=page_number
                )
            )
        
        return results
    
    @classmethod
    def list_documents(
        cls
        ) -> list[dict[str, object]]:
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
    def delete_document(
        cls,
        document_id: int
        ) -> None:
        with cls.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s;
                    """,
                    (document_id,)
                )
  
