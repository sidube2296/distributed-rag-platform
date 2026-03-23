from sentence_transformers import CrossEncoder

# Load reranker model
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=2):

    pairs = [[query, doc] for doc in documents]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(documents, scores))

    # Sort by score (highest first)
    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Return top_k documents
    return [doc for doc, score in ranked[:top_k]]
