"""
Generator module for the Carbon Accounting Chatbot.

This module defines two chatbot implementations:
1. CarbonChatbot: Uses local Hugging Face models (BART).
2. AzureCarbonChatbot: Uses Azure OpenAI for generation.
"""
import json
import os
import sys

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

# Force the system to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Limit threading to prevent crash on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Configuration
INDEX_PATH = "data/vector_db/carbon_index.faiss"
METADATA_PATH = "data/vector_db/metadata.json"
# Using BART-large for high-quality generation
GEN_MODEL = "facebook/bart-large-cnn"
EMBED_MODEL = "all-MiniLM-L6-v2"


class CarbonChatbot:
    """Chatbot using local Hugging Face models."""

    def __init__(self):
        """Initializes the retriever and generator models."""
        # Load the Retriever (FAISS + Embedder)
        self.index = faiss.read_index(INDEX_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)

        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        # Load the Generator (BART)
        print("Loading Generator model...")
        self.generator = pipeline("summarization", model=GEN_MODEL)

    def ask(self, query, top_k=3):
        """Retrieves context and generates an answer."""
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

        # Step D: Generate the final answer
        prompt = f"Using the following carbon accounting guidance, answer the question: {query}\n\nGuidance: {retrieved_context}"

        # Use 'summarization' pipeline to synthesize the answer
        result = self.generator(prompt, max_length=150, min_length=30, do_sample=False)
        answer = result[0]["summary_text"]

        return {
            "answer": answer,
            "citations": list(set(sources)),  # Unique sources only
        }


class AzureCarbonChatbot:
    """Chatbot using Azure OpenAI for generation."""

    def __init__(self):
        """Initializes Azure OpenAI client and local retriever."""
        # Use the Azure-specific class
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Load local search components
        self.index = faiss.read_index("data/vector_db/carbon_index.faiss")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        with open("data/vector_db/metadata.json", "r") as f:
            self.metadata = json.load(f)

    def ask(self, query, top_k=5):
        """Retrieves context and generates an answer using Azure OpenAI."""
        # Local Semantic Search
        query_vec = self.embedder.encode([query]).astype("float32")
        _, indices = self.index.search(query_vec, top_k)

        # Build Context
        context_parts = []
        source_list = []
        for i in indices[0]:
            chunk = self.metadata[i]
            context_parts.append(chunk["text"])
            source_list.append(
                f"{chunk['metadata']['source']} (Pg {chunk['metadata']['page']})"
            )

        context_text = " ".join(context_parts)

        # Generation
        completion = self.client.chat.completions.create(
            model=self.deployment_name,  # Matches Azure Deployment Name
            messages=[
                {
                    "role": "system",
                    "content": "You are a Carbon Accounting Expert. Use the provided context to answer accurately.",
                },
                {
                    "role": "user",
                    "content": f"Context: {context_text}\n\nQuestion: {query}",
                },
            ],
            temperature=0.7,
        )

        return {
            "answer": completion.choices[0].message.content,
            "citations": list(set(source_list)),
        }


if __name__ == "__main__":
    # Example usage
    # bot = CarbonChatbot()
    bot = AzureCarbonChatbot()
    user_query = "What are the core requirements for SECR reporting?"
    response = bot.ask(user_query)

    print(f"\nAI Answer: {response['answer']}")
    print("\nSources:")
    for src in response["citations"]:
        print(f"- {src}")
