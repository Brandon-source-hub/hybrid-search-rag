class HybridRetriever:

    def __init__(self, dense_retriever, bm25_retriever, embedding_model):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model

    def search(self, query, top_k=5, candidate_k=10, rrf_k=60):

        query_embedding = self.embedding_model.encode_query(query)

        dense_results = self.dense_retriever.search(
            query_embedding,
            top_k=candidate_k
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k
        )

        fused_scores = {}
        result_map = {}

        # Dense ranking
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            result_map[chunk_id] = result

        # BM25 ranking
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            result_map[chunk_id] = result

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )

        results = []

        for chunk_id in ranked_ids[:top_k]:
            result = result_map[chunk_id].copy()
            result["rrf_score"] = fused_scores[chunk_id]
            results.append(result)

        return results