from rank_bm25 import BM25Okapi

def create_bm25_index(chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, tokenized_chunks


def bm25_search(query, bm25, tokenized_chunks, chunks, top_k=2):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = [chunks[i] for i in top_indices]
    return results
