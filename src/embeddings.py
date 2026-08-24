from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode_documents(self,texts):
        embeddings=self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    def encode_query(self,query):
        embedding=self.model.encode([query], normalize_embeddings=True)
        return embedding