"""
Retriever module for building a FAISS vector index.

This script loads processed text chunks, generates embeddings using a transformer model,
and builds a FAISS index for efficient similarity search.
"""
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
CHUNKS_PATH = "data/processed/processed_chunks.json"
INDEX_PATH = "data/vector_db/carbon_index.faiss"
METADATA_PATH = "data/vector_db/metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # Light and fast, perfect for low latency

os.makedirs("data/vector_db/", exist_ok=True)


def build_index():
    """Builds and saves the FAISS index and metadata."""
    # 1. Load the processed chunks
    with open(CHUNKS_PATH, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]

    # 2. Initialize the embedding model
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Create embeddings (convert text to numbers)
    print("Generating embeddings for chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 4. Build the FAISS index for fast similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 5. Save the index and the metadata
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(data, f)

    print(f"Success. Vector DB built with {len(texts)} chunks.")


if __name__ == "__main__":
    build_index()
