
from vector_store import NeonVectorStore

document_id = 1

NeonVectorStore.delete_document(document_id)
print(f"Deleted document with ID: {document_id}")