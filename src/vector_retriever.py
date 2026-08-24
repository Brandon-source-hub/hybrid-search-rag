import faiss
import numpy as np

class VectorRetriever:
    def __init__(self,embeddings,chunks):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(
            np.array(embeddings, dtype=np.float32)
        )
        self.chunks = chunks

    def search(self, query_embedding, top_k=5):
        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype=np.float32), top_k
        )

        results = []
        for score,idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            results.append(
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "score": float(score)
                }
            )
        return results