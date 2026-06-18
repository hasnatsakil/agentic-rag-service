
from vector_store import NeonVectorStore

documents = NeonVectorStore.list_documents()

print("Documents:")

for doc in documents:
    print(f"{doc['id']}: {doc['file_name']} {doc['created_at']}")