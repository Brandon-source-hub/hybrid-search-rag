# Hybrid Search RAG

A production-oriented Retrieval-Augmented Generation project exploring dense retrieval, BM25, hybrid search, reranking and retrieval evaluation.

## Current Features

- PDF and TXT document ingestion
- Text chunking with overlap
- Sentence Transformer embeddings
- FAISS vector search
- Top-K semantic retrieval
- Source tracking

## Architecture

```
Documents
    ↓
Text Extraction
    ↓
Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
Top-K Relevant Chunks
```

## Project Structure

```
hybrid_search_rag/
├── main.py                  # Interactive CLI entry point
├── requirements.txt
├── data/
│   └── documents/           # Place PDF / TXT files here
└── src/
    ├── ingestion.py         # Document loading (PDF, TXT)
    ├── chunking.py          # Text chunking with overlap
    ├── embeddings.py        # Sentence Transformer wrapper
    └── vector_retriever.py  # FAISS-based dense retrieval
```

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

1. Drop your `.pdf` or `.txt` files into `data/documents/`.
2. Run the interactive retriever:

```bash
python main.py
```

3. Enter a question at the prompt. The top 5 most relevant chunks are printed with their similarity score and source file. Type `exit` to quit.

## Roadmap

- [x] Dense retrieval baseline
- [ ] BM25 retrieval
- [ ] Hybrid retrieval
- [ ] Reciprocal Rank Fusion
- [ ] Cross-encoder reranking
- [ ] Retrieval evaluation
- [ ] FastAPI backend
- [ ] Docker deployment
