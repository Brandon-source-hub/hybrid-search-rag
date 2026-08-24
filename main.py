from src.ingestion import load_documents 
from src.chunking import chunk_documents 
from src.embeddings import EmbeddingModel 
from src.vector_retriever import VectorRetriever 
DATA_PATH = "data/documents" 
def main(): 
    print("Loading documents...") 
    documents = load_documents(DATA_PATH) 
    print( f"Loaded {len(documents)} documents." ) 
    chunks = chunk_documents(documents) 
    print( f"Created {len(chunks)} chunks." ) 
    embedding_model = EmbeddingModel() 
    texts = [ chunk["text"] for chunk in chunks ] 
    print("Creating embeddings...") 
    embeddings = embedding_model.encode_documents( texts ) 
    retriever = VectorRetriever( 
        embeddings, chunks
        ) 
    print("\nRAG Retriever ready.") 
    while True: 
        query = input( "\nEnter your question (or 'exit'): " ) 
        if query.lower() == "exit": break 
        query_embedding = ( embedding_model.encode_query(query) ) 
        results = retriever.search( query_embedding, top_k=5 ) 
        print("\nTop results:\n") 
        for i, result in enumerate( results, start=1 ): 
            print( f"Result {i}" ) 
            print( f"Score: {result['score']:.4f}" )
            print( f"Source: {result['source']}" ) 
            print( result["text"][:500] ) 
            print( "-" * 80 ) 

if __name__ == "__main__": 
    main()