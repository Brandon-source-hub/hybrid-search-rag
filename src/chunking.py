def chunk_text(text, chunk_size=500,overlap=50):
    """
    Splits the input text into chunks of specified size with optional overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk. Default is 500.
        overlap (int): The number of overlapping characters between chunks. Default is 50.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start index forward by chunk_size - overlap
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents):
    all_chunks = []
    for document in documents:
        chunks = chunk_text(document["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "source": document["source"],
                    "chunk_index": i,
                    "text": chunk
                }
            )
    return all_chunks