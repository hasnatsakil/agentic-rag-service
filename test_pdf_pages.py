from document_loader import DocumentLoader

pages = DocumentLoader.load_pdf_pages("/home/sakil/Documents/LLM Learning/MMU_Guideline-of-thesis-preparation_V4.2_July2025.pdf")

print("pages with text:", len(pages))

for page in pages:
    print("Page:", page["page_number"])
    print(page["text"][:300])
    print("---")