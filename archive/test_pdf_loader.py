
from document_loader import DocumentLoader

text = DocumentLoader.load_pdf("/home/sakil/Documents/LLM Learning/test_rag/sample.pdf")

print("Characters extracted", len(text))
print()
print(text[:500])  # Print the first 500 characters