from sentence_transformers import CrossEncoder

# Load model once
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=2):
    pairs = [[query, doc] for doc in documents]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(documents, scores))

    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]