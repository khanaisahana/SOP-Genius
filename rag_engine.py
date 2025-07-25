# Build ChromaDB RAG Engine

import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

EMBED_MODEL = "all-MiniLM-L6-v2"
persist_dir = "chromadb_store"

client = chromadb.PersistentClient(path=persist_dir)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

collection = client.get_or_create_collection(name="sop_docs", embedding_function=embedding_func)

def load_txt_to_chromadb(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    doc_id = os.path.basename(filepath)
    collection.add(documents=[content], ids=[doc_id])

def query_sop_docs(user_query, top_k=3):
    results = collection.query(query_texts=[user_query], n_results=top_k)
    if results["documents"]:
        return "\n\n".join(doc[0] for doc in results["documents"])
    return "No relevant SOP found."
