"""
Processor module for extracting text from PDFs and creating chunks.

This script reads raw PDF files, cleans the text, splits it into manageable chunks,
and saves the processed data to a JSON file for embedding.
"""
import json
import os

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
RAW_DIR = "data/raw/"
PROCESSED_FILE = "data/processed/processed_chunks.json"
os.makedirs("data/processed/", exist_ok=True)

# Chunking parameters (approx 150-300 words = 1000-2000 characters)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
)


def process_pdfs():
    """Extracts text from PDFs, chunks it, and saves to JSON."""
    all_chunks = []

    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs. Starting extraction...")

    for file_name in pdf_files:
        file_path = os.path.join(RAW_DIR, file_name)

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue

                    # Remove excess whitespace
                    clean_text = " ".join(text.split())

                    chunks = text_splitter.split_text(clean_text)

                    for chunk in chunks:
                        all_chunks.append(
                            {
                                "text": chunk,
                                "metadata": {
                                    "source": file_name,
                                    "page": i + 1,
                                    "type": "Carbon Regulatory Document",
                                },
                            }
                        )
            print(f"Successfully processed: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Save chunks for embedding
    with open(PROCESSED_FILE, "w") as f:
        json.dump(all_chunks, f, indent=4)

    print(f"\nProcessing complete. Created {len(all_chunks)} total chunks.")
    print(f"Data saved to {PROCESSED_FILE}")


if __name__ == "__main__":
    process_pdfs()
