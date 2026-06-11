
from vector_store import NeonVectorStore

documents = NeonVectorStore.list_documents()

print("Documents:")

for document_id, file_name, created_at in documents:
    print(f"{document_id}: {file_name} {created_at}")