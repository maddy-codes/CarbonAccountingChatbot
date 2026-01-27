# Carbon Accounting Chatbot

An intelligent chatbot designed to assist with carbon accounting covering SECR, GHG Protocol, and other regulatory frameworks. This application utilizes Retrieval-Augmented Generation (RAG) to provide accurate answers sourced directly from official guidance documents.

## Features

- **Automated Data Ingestion**: Scrapes and processes PDF documents from major regulatory bodies (UK Gov, EPA, GHG Protocol).
- **Semantic Search**: Uses FAISS and Sentence Transformers to retrieve relevant context.
- **AI-Powered Answers**: Generates responses using Azure OpenAI (or local models) based on retrieved regulations.
- **Interactive UI**: User-friendly chat interface built with Streamlit.
- **Source Citation**: Every response includes citations to the specific documents and pages used.

## Project Structure

```
CarbonAccountingChatbot/
├── app/
│   └── main.py          # Streamlit application entry point
├── data/
│   ├── raw/             # Downloaded PDF documents
│   ├── processed/       # Extracted text chunks
│   └── vector_db/       # FAISS index and metadata
├── src/
│   ├── ingestion/
│   │   ├── scraper.py   # Web scraper for regulations
│   │   └── processor.py # Text extraction and chunking
│   └── models/
│       ├── generator.py # RAG generation logic
│       └── retriever.py # Vector database management
├── .env.example         # Environment variable template
└── README.md            # Project documentation
```

## Installation

This project uses `uv` for fast dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/CarbonAccountingChatbot.git
    cd CarbonAccountingChatbot
    ```

2.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Install dependencies**:
    Initialize the environment and install packages using `uv`:
    ```bash
    uv sync
    ```

4.  **Set up environment variables**:
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```

## Usage

1.  **Data Models Initialization** (First run only):
    Download regulations, process text, and build the vector index.
    ```bash
    uv run src/ingestion/scraper.py
    uv run src/ingestion/processor.py
    uv run src/models/retriever.py
    ```

2.  **Run the Application**:
    Start the Streamlit interface.
    ```bash
    uv run streamlit run app/main.py
    ```

## Technology Stack

- **Python**
- **Streamlit**: Web Interface
- **FAISS**: Vector Database
- **Sentence Transformers**: Text Embeddings
- **Azure OpenAI**: Language Model
- **LangChain**: Text Processing
