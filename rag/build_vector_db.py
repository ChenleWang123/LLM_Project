import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Define paths
CORPUS_FILE = "../data/corpus/master_knowledge.txt"
VECTOR_STORE_DIR = "../vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_STORE_DIR, "meta.json")

# Use a lightweight multi-lingual model
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_db():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # Read the corpus line by line
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]

    print(f"Loading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Encoding {len(chunks)} chunks into vectors...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Initialize FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save index and text mapping
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print("✅ Vector database built successfully!")


if __name__ == "__main__":
    build_db()