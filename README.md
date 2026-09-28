# 📚 Document Intelligence RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) application that lets users upload technical PDF documents and ask questions grounded in those documents.

## Features

- Upload multiple PDF knowledge sources
- Extract text and page metadata using PyPDF
- Clean and split documents into overlapping chunks
- Generate semantic embeddings with Gemini
- Store embeddings in a local FAISS vector index
- Retrieve the most relevant document chunks for each question
- Generate grounded answers with Gemini through LangChain
- Display source file, page number, and retrieval score
- Refuse to fabricate an answer when the information is not found in the uploaded context
- Streamlit interface for an easy demonstration

## Architecture

```text
             ┌──────────────────┐
             │   PDF Documents  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  PDF Extraction │
             │     + Cleanup   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Chunking +       │
             │ Metadata         │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Gemini Embedding │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │   FAISS Index    │
             └────────┬─────────┘
                      │
            User Question
                      │
                      ▼
             ┌──────────────────┐
             │ Semantic Search  │
             │   Top-K Chunks   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ LangChain Prompt │
             │ + Gemini LLM     │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Grounded Answer  │
             │ + Sources        │
             └──────────────────┘
```

## Tech Stack

- Python
- LangChain
- Google Gemini API
- Gemini Embeddings
- FAISS
- PyPDF
- Streamlit
- python-dotenv

## Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd document-intelligence-rag
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Then edit `.env`:

```text
GEMINI_API_KEY=your_actual_key
```

Never commit `.env` or your API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## How to use

1. Upload one or more technical PDFs.
2. Click **Build / Rebuild Knowledge Base**.
3. Wait for extraction, chunking, embedding and FAISS indexing.
4. Ask a question in the chat box.
5. The system retrieves relevant chunks.
6. Gemini generates an answer using only the retrieved context.
7. Expand **Sources** to see the supporting PDF and page numbers.

## Example questions

- What are the main components described in the document?
- Explain the authentication flow.
- What are the system requirements?
- What are the key steps in the installation process?
- Which configuration parameters are required?
- What limitations are mentioned?

## Project Structure

```text
document-intelligence-rag/
│
├── app.py                 # Streamlit UI
├── rag_pipeline.py        # RAG ingestion, retrieval and generation
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/                  # Uploaded PDFs (ignored by Git)
│
└── vectorstore/           # Generated FAISS index (ignored by Git)
```

## Important implementation details

### Chunking
Documents are split into approximately 1000-character chunks with 150-character overlap. Overlap helps preserve context across chunk boundaries.

### Retrieval
FAISS performs vector similarity search over Gemini-generated embeddings and returns the top-K relevant chunks.

### Grounding
The generation prompt explicitly restricts the model to the retrieved context and instructs it to say when the answer is not available.

### Source awareness
Each chunk preserves the original PDF filename and page number, allowing the UI to expose supporting document locations.

## Future improvements

- Hybrid BM25 + vector retrieval
- Reranking retrieved chunks
- Conversational memory
- OCR for scanned PDFs
- User authentication
- Evaluation with RAGAS
- Retrieval caching
- Cloud vector database
- Docker deployment
- Streaming responses
