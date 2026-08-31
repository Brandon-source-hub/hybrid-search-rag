import json
from pathlib import Path

from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_retriever import VectorRetriever
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker


DATA_PATH = "data/documents"
QUERY_PATH = "evaluation/queries.json"


# =========================================================
# Metrics
# =========================================================

def hit_at_k(results, relevant_chunks, k):

    retrieved_ids = [
        result["chunk_id"]
        for result in results[:k]
    ]

    for relevant_chunk in relevant_chunks:
        if relevant_chunk in retrieved_ids:
            return 1

    return 0


def recall_at_k(results, relevant_chunks, k):

    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    relevant_set = set(relevant_chunks)

    if len(relevant_set) == 0:
        return 0.0

    retrieved_relevant = (
        retrieved_ids.intersection(relevant_set)
    )

    return (
        len(retrieved_relevant)
        / len(relevant_set)
    )


def reciprocal_rank(results, relevant_chunks):

    relevant_set = set(relevant_chunks)

    for rank, result in enumerate(
        results,
        start=1
    ):

        if result["chunk_id"] in relevant_set:

            return 1 / rank

    return 0.0


# =========================================================
# Evaluation
# =========================================================

def evaluate_retriever(
    name,
    queries,
    search_function,
    top_k=5
):

    hit1_scores = []
    hit3_scores = []
    hit5_scores = []

    recall_scores = []
    rr_scores = []

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Evaluating: {name}"
    )

    print(
        f"{'=' * 60}"
    )

    for item in queries:

        query = item["query"]

        relevant_chunks = (
            item["relevant_chunks"]
        )

        results = search_function(
            query,
            top_k
        )

        hit1 = hit_at_k(
            results,
            relevant_chunks,
            1
        )

        hit3 = hit_at_k(
            results,
            relevant_chunks,
            3
        )

        hit5 = hit_at_k(
            results,
            relevant_chunks,
            5
        )

        recall5 = recall_at_k(
            results,
            relevant_chunks,
            5
        )

        rr = reciprocal_rank(
            results,
            relevant_chunks
        )

        hit1_scores.append(hit1)
        hit3_scores.append(hit3)
        hit5_scores.append(hit5)

        recall_scores.append(recall5)
        rr_scores.append(rr)

        print(
            f"\nQuery: {query}"
        )

        print(
            f"Relevant: {relevant_chunks}"
        )

        retrieved = [
            result["chunk_id"]
            for result in results[:top_k]
        ]

        print(
            f"Retrieved: {retrieved}"
        )

        print(
            f"Hit@1={hit1}, "
            f"Hit@3={hit3}, "
            f"Hit@5={hit5}, "
            f"Recall@5={recall5:.2f}, "
            f"RR={rr:.3f}"
        )

    number_queries = len(queries)

    metrics = {

        "Hit@1":
            sum(hit1_scores)
            / number_queries,

        "Hit@3":
            sum(hit3_scores)
            / number_queries,

        "Hit@5":
            sum(hit5_scores)
            / number_queries,

        "Recall@5":
            sum(recall_scores)
            / number_queries,

        "MRR":
            sum(rr_scores)
            / number_queries
    }

    return metrics


# =========================================================
# Main
# =========================================================

def main():

    # ---------------------------------------------
    # Load evaluation queries
    # ---------------------------------------------

    with open(
        QUERY_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        queries = json.load(f)

    # ---------------------------------------------
    # Load documents
    # ---------------------------------------------

    print("Loading documents...")

    documents = load_documents(
        DATA_PATH
    )

    chunks = chunk_documents(
        documents
    )

    print(
        f"Loaded {len(documents)} documents"
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    # ---------------------------------------------
    # Embeddings
    # ---------------------------------------------

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = (
        embedding_model
        .encode_documents(texts)
    )

    # ---------------------------------------------
    # Retrievers
    # ---------------------------------------------

    dense_retriever = VectorRetriever(
        embeddings,
        chunks
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        embedding_model
    )

    reranker = (
        CrossEncoderReranker()
    )

    # ---------------------------------------------
    # Search wrappers
    # ---------------------------------------------

    def dense_search(
        query,
        top_k
    ):

        query_embedding = (
            embedding_model
            .encode_query(query)
        )

        return (
            dense_retriever
            .search(
                query_embedding,
                top_k=top_k
            )
        )

    def bm25_search(
        query,
        top_k
    ):

        return (
            bm25_retriever
            .search(
                query,
                top_k=top_k
            )
        )

    def hybrid_search(
        query,
        top_k
    ):

        return (
            hybrid_retriever
            .search(
                query,
                top_k=top_k
            )
        )

    def reranked_search(
        query,
        top_k
    ):

        # First retrieve more candidates
        candidates = (
            hybrid_retriever
            .search(
                query,
                top_k=8
            )
        )

        # Then rerank
        return (
            reranker
            .rerank(
                query,
                candidates,
                top_k=top_k
            )
        )

    # ---------------------------------------------
    # Evaluate all retrieval methods
    # ---------------------------------------------

    all_results = {}

    all_results["Dense"] = (
        evaluate_retriever(
            "Dense",
            queries,
            dense_search
        )
    )

    all_results["BM25"] = (
        evaluate_retriever(
            "BM25",
            queries,
            bm25_search
        )
    )

    all_results["Hybrid RRF"] = (
        evaluate_retriever(
            "Hybrid RRF",
            queries,
            hybrid_search
        )
    )

    all_results[
        "Hybrid + Reranker"
    ] = evaluate_retriever(
        "Hybrid + Reranker",
        queries,
        reranked_search
    )

    # ---------------------------------------------
    # Final summary
    # ---------------------------------------------

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 80
    )

    header = (
        f"{'Method':<22}"
        f"{'Hit@1':<10}"
        f"{'Hit@3':<10}"
        f"{'Hit@5':<10}"
        f"{'Recall@5':<12}"
        f"{'MRR':<10}"
    )

    print(header)

    print("-" * 80)

    for method, metrics in (
        all_results.items()
    ):

        print(
            f"{method:<22}"
            f"{metrics['Hit@1']:<10.3f}"
            f"{metrics['Hit@3']:<10.3f}"
            f"{metrics['Hit@5']:<10.3f}"
            f"{metrics['Recall@5']:<12.3f}"
            f"{metrics['MRR']:<10.3f}"
        )


if __name__ == "__main__":
    main()