from dataclasses import dataclass

@dataclass
class RetrievalResult:
    score: float
    chunk_id: int
    chunk_text: str
    page_number: int | None = None

    def label(
            self
        ) -> str:
         if self.page_number:
             return f"Page {self.page_number}, Chunk {self.chunk_id + 1}"
         
         return f"Chunk {self.chunk_id + 1}"


@dataclass
class IngestResult:
    DOCUMENT_ID: int
    file_name: str
    page_count: int
    chunk_count: int

@dataclass
class ChatResult:
    answer: str
    sources: list[RetrievalResult]

