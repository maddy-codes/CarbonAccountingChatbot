import os
import sys

import torch

# 1. Force the system to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 2. Limit threading to prevent the crash on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- CONFIGURATION ---
INDEX_PATH = "data/vector_db/carbon_index.faiss"
METADATA_PATH = "data/vector_db/metadata.json"
# Using BART-large for high-quality generation as per your proposal
GEN_MODEL = "facebook/bart-large-cnn"
EMBED_MODEL = "all-MiniLM-L6-v2"


class CarbonChatbot:
    def __init__(self):
        # 1. Load the Retriever (FAISS + Embedder)
        self.index = faiss.read_index(INDEX_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)

        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        # 2. Load the Generator (BART) [cite: 191]
        print("Loading Generator model...")
        self.generator = pipeline("summarization", model=GEN_MODEL)

    def ask(self, query, top_k=3):
        # Step A: Embed the question
        query_vector = self.embedder.encode([query]).astype("float32")

        # Step B: Search the index for the best chunks
        distances, indices = self.index.search(query_vector, top_k)

        # Step C: Collect context and citations
        retrieved_context = ""
        sources = []

        for idx in indices[0]:
            chunk = self.metadata[idx]
            retrieved_context += chunk["text"] + " "
            sources.append(
                f"{chunk['metadata']['source']} (Page {chunk['metadata']['page']})"
            )

        # Step D: Generate the final answer [cite: 105]
        prompt = f"Using the following carbon accounting guidance, answer the question: {query}\n\nGuidance: {retrieved_context}"

        # We use 'summarization' pipeline as a way to synthesize the answer
        result = self.generator(prompt, max_length=150, min_length=30, do_sample=False)
        answer = result[0]["summary_text"]

        return {
            "answer": answer,
            "citations": list(set(sources)),  # Unique sources only
        }


if __name__ == "__main__":
    bot = CarbonChatbot()
    user_query = "What are the core requirements for SECR reporting?"
    response = bot.ask(user_query)

    print(f"\nAI Answer: {response['answer']}")
    print("\nSources:")
    for src in response["citations"]:
        print(f"- {src}")
