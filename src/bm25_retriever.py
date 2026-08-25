from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self,chunks):
        self.chunks = chunks
        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, top_k=5):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]

            results.append({
                "score": float(scores[idx]),
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk.get(
                    "chunk_id",
                    f"{chunk['source']}:{chunk['chunk_index']}"
                )
            })

        return results
    
