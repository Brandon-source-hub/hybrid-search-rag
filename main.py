from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_retriever import VectorRetriever
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever

DATA_PATH = "data/documents"


def main():

    print("Loading documents...")

    # 1. Load documents
    documents = load_documents(DATA_PATH)

    print(f"Loaded {len(documents)} documents.")

    # 2. Chunk documents
    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # 3. Create embedding model
    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # 4. Create dense embeddings
    print("Creating embeddings...")

    embeddings = embedding_model.encode_documents(
        texts
    )

    # 5. Create Dense Retriever
    dense_retriever = VectorRetriever(
        embeddings,
        chunks
    )

    # 6. Create BM25 Retriever
    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
    dense_retriever,
    bm25_retriever,
    embedding_model
    )

    print("\nRetrievers ready.")

    # 7. Query loop
    while True:

        query = input(
            "\nEnter your question (or 'exit'): "
        )

        if query.lower() == "exit":
            break

        # -----------------------------
        # Dense Retrieval
        # -----------------------------

        query_embedding = embedding_model.encode_query(
            query
        )

        dense_results = dense_retriever.search(
            query_embedding,
            top_k=5
        )

        # -----------------------------
        # BM25 Retrieval
        # -----------------------------

        bm25_results = bm25_retriever.search(
            query,
            top_k=5
        )

        # -----------------------------
        # Hybrid Retrieval
        # -----------------------------

        hybrid_results = hybrid_retriever.search(
            query,
            top_k=5
        )
        # -----------------------------
        # Print Dense Results
        # -----------------------------

        print("\n===== Dense Retrieval =====\n")

        for i, result in enumerate(
            dense_results,
            start=1
        ):

            print(f"Result {i}")

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print(
                result["text"][:500]
            )

            print("-" * 80)

        # -----------------------------
        # Print BM25 Results
        # -----------------------------

        print("\n===== BM25 Retrieval =====\n")

        for i, result in enumerate(
            bm25_results,
            start=1
        ):

            print(f"Result {i}")

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print(
                result["text"][:500]
            )

            print("-" * 80)

        # -----------------------------
        # Print Hybrid Results
        # -----------------------------
        

        print("\n===== Hybrid Retrieval (RRF) =====\n")

        for i, result in enumerate(hybrid_results, start=1):

            print(f"Result {i}")
            print(f"RRF Score: {result['rrf_score']:.6f}")
            print(f"Source: {result['source']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(result["text"][:500])
            print("-" * 80)


if __name__ == "__main__":
    main()