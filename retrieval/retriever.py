from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, index, chunks):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), 3)

    results = [chunks[i] for i in indices[0]]

    return results