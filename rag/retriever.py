import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = "vector_store"  # Assuming run from project root
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_STORE_DIR, "meta.json")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class CulturalRetriever:
    def __init__(self):
        """Load the pre-built FAISS index and embedding model."""
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query, top_k=3):
        """
        Search for the most relevant cultural facts based on the query.
        Returns a single concatenated string of background context.
        """
        # Convert user query to vector
        query_vec = self.model.encode([query])

        # Search FAISS
        distances, indices = self.index.search(np.array(query_vec).astype("float32"), top_k)

        # Collect results
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])

        return "\n".join(results)


# Quick test
if __name__ == "__main__":
    retriever = CulturalRetriever()
    test_q = "What is calligraphy called in China?"
    print("Test Query:", test_q)
    print("Retrieved Context:\n", retriever.search(test_q))