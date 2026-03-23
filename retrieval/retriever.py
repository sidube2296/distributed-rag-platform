from sentence_transformers import SentenceTransformer
import numpy as np
from retrieval.bm25_retriever import bm25_search

model = SentenceTransformer("all-MiniLM-L6-v2")

def hybrid_search(query, index, chunks, bm25, tokenized_chunks, top_k=2):

    # VECTOR SEARCH
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    vector_results = [chunks[i] for i in indices[0]]

    # BM25 SEARCH
    keyword_results = bm25_search(query, bm25, tokenized_chunks, chunks, top_k)

    # COMBINE RESULTS
    combined = list(set(vector_results + keyword_results))

    return combined