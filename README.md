# Hybrid Search RAG

A production-oriented Retrieval-Augmented Generation project exploring dense retrieval, BM25, hybrid search, cross-encoder reranking and retrieval evaluation.

## Current Features

- PDF and TXT document ingestion
- Text chunking with overlap
- Sentence Transformer embeddings
- FAISS vector search (dense)
- BM25 sparse retrieval (rank-bm25)
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Retrieval evaluation (Hit@k, Recall@k, MRR)

## Architecture

```
Documents
    ↓
Text Extraction
    ↓
Chunking
    ↓
┌─────────────────────┬─────────────────────┐
│  Dense:             │  Sparse:             │
│  Sentence Transformer│  BM25 (rank-bm25)   │
│  → FAISS index      │  → tokenized corpus │
└─────────────────────┴─────────────────────┘
            ↓
    Hybrid Retrieval (RRF fusion)
            ↓
    Cross-encoder Reranking
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
├── evaluation/
│   ├── evaluate.py          # Retrieval evaluation (Hit@k, Recall@k, MRR)
│   └── queries.json         # Ground-truth query/relevant-chunk pairs
└── src/
    ├── ingestion.py         # Document loading (PDF, TXT)
    ├── chunking.py          # Text chunking with overlap
    ├── embeddings.py        # Sentence Transformer wrapper
    ├── vector_retriever.py  # FAISS-based dense retrieval
    ├── bm25_retriever.py    # BM25 sparse retrieval
    ├── hybrid_retriever.py  # RRF fusion of dense + sparse
    └── reranker.py          # Cross-encoder reranking
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

3. Enter a question at the prompt. Dense, BM25, hybrid and reranked results are printed with their score and source file. Type `exit` to quit.

## Evaluation

Compare retrieval methods against ground-truth queries:

```bash
python evaluation/evaluate.py
```

Reports Hit@1, Hit@3, Hit@5, Recall@5 and MRR for Dense, BM25, Hybrid RRF and Hybrid + Reranker. Ground-truth relevance pairs are defined in `evaluation/queries.json`.

## Roadmap

- [x] Dense retrieval baseline
- [x] BM25 sparse retrieval
- [x] Hybrid retrieval
- [x] Reciprocal Rank Fusion
- [x] Cross-encoder reranking
- [x] Retrieval evaluation
- [ ] LLM generation
- [ ] FastAPI backend
- [ ] Docker deployment
