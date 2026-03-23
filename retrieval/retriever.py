from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, index, chunks, top_k=2):

    # Convert query → embedding
    query_embedding = model.encode([query])

    # Search similar vectors
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Get matching chunks
    results = [chunks[i] for i in indices[0] if distances[0][list(indices[0]).index(i)] < 1.5]

    return results