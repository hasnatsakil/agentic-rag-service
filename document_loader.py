
from pathlib import Path

from pypdf import PdfReader

class DocumentLoader:

    @staticmethod
    def load_pdf(
        file_path: str
        ) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        reader = PdfReader(path)

        page_text = []

        for page_number, page in enumerate(reader.pages, start= 1):
            text = page.extract_text()

            if text:
                page_text.append(text)

        return "\n".join(page_text)
    
    @staticmethod
    def load_pdf_pages(
        file_path: str
        ) -> list[dict]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        reader = PdfReader(path)

        pages = []

        for page_number, page in enumerate(reader.pages, start= 1):
            text = page.extract_text()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

        return pages
    