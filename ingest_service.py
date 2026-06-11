
from pathlib import Path

from chunking import RecursiveChunk
from rag_engine import RAGEngine
from document_loader import DocumentLoader
from models import IngestResult
from vector_store import NeonVectorStore

class IngestService:
    @staticmethod
    def ingest_pdf(
        file_path: str
        ) -> IngestResult:
        path = Path(file_path)

        pages = DocumentLoader.load_pdf_pages(file_path)

        if not pages:
            raise ValueError("No extractable text found in PDF.")

        all_chunks = []
        page_numbers = []

        for page in pages:
            page_chunks = RecursiveChunk.chunk(page["text"])

            for chunk in page_chunks:
                all_chunks.append(chunk)
                page_numbers.append(page["page_number"])
        if not all_chunks:
            raise ValueError("No chunks created from PDF text.")
        
        embeddings = RAGEngine.embed_chunks(all_chunks)

        document_id = NeonVectorStore.create_document(file_name = path.name)

        NeonVectorStore.insert_chunks(
        document_id = document_id,
        chunks = all_chunks,
        embeddings = embeddings,
        page_numbers = page_numbers
        )

        return IngestResult(
            document_id=document_id,
            file_name=path.name,
            page_count=len(pages),
            chunk_count=len(all_chunks)
        )
