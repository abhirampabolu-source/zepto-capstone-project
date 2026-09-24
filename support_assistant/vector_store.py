import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

def initialize_vector_store():
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name="zepto_policies",
        embedding_function=sentence_transformer_ef
    )
    
    if os.path.exists(DOCS_DIR):
        documents = []
        metadatas = []
        ids = []
        
        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename.startswith("doc_") and filename.endswith(".txt"):
                filepath = os.path.join(DOCS_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                documents.append(content)
                metadatas.append({"source": filename})
                ids.append(filename)
        
        if documents:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully indexed {len(documents)} policy documents into ChromaDB.")
            
    return collection

if __name__ == "__main__":
    initialize_vector_store()
