import sys

from ingest_service import IngestService


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    result = IngestService.ingest_pdf(file_path)

    print("Ingested file:", result.file_name)
    print("Document id:", result.document_id)
    print("Pages:", result.page_count)
    print("Chunks:", result.chunk_count)