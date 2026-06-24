
from pathlib import Path

from core.chunking import RecursiveChunk
from core.rag_engine import RAGEngine
from core.document_loader import DocumentLoader
from core.models import IngestResult
from core.vector_store import NeonVectorStore

class IngestService:
    @staticmethod
    def ingest_pdf(
        file_path: str,
        original_file_name: str | None = None
        ) -> IngestResult:
        path = Path(file_path)
        file_name = original_file_name if original_file_name else path.name

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

        DOCUMENT_ID = NeonVectorStore.create_document(file_name = file_name)

        NeonVectorStore.insert_chunks(
        DOCUMENT_ID = DOCUMENT_ID,
        chunks = all_chunks,
        embeddings = embeddings,
        page_numbers = page_numbers
        )

        return IngestResult(
            DOCUMENT_ID=DOCUMENT_ID,
            file_name=file_name,
            page_count=len(pages),
            chunk_count=len(all_chunks)
        )
