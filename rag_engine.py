
# rag_engine.py

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBED_MODEL)

# Paths
INDEX_PATH = "faiss_index/index.faiss"
DOCS_PATH = "faiss_index/docs.pkl"
os.makedirs("faiss_index", exist_ok=True)

# Load or create FAISS index
dimension = embedding_model.get_sentence_embedding_dimension()
if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        doc_store = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    doc_store = []  # Stores the original documents

def load_txt_to_faiss(filepath):  # Retained function name for compatibility
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    embedding = embedding_model.encode([content])
    index.add(embedding)
    doc_store.append(content)

    # Persist index and docs
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(doc_store, f)

def query_sop_docs(user_query, top_k=3):
    if index.ntotal == 0:
        return "No documents available."

    query_embedding = embedding_model.encode([user_query])
    distances, indices = index.search(query_embedding, top_k)

    matched_docs = []
    for idx in indices[0]:
        if idx < len(doc_store):
            matched_docs.append(doc_store[idx])
    return "\n\n".join(matched_docs) if matched_docs else "No relevant SOP found."
