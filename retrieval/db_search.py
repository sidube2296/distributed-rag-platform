from sentence_transformers import SentenceTransformer
import psycopg2
from retrieval.query_classifier import classify_query
from retrieval.reranker import rerank

# 🔥 DEFINE MODEL (this was missing)
model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    dbname="rag_db",
    user="siddhi2218",
    password="",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()


def db_search(query, top_k=20):

    domain = classify_query(query)

    expanded_query = query + " transformer attention neural network NLP BERT GPT"

    query_embedding = str(model.encode(expanded_query).tolist())

    cursor.execute(
        """
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_embedding, top_k)
    )

    results = cursor.fetchall()
    docs = [r[0] for r in results]
	
    print("\n--- RAW RESULTS ---")
    for d in docs[:5]:
    	print(d[:200])

    # 🔥 DOMAIN FILTER
    if domain == "nlp":
        docs = [
            d for d in docs
            if "transformer" in d.lower() or "attention" in d.lower()
        ]

    # fallback
    if len(docs) < 2:
        docs = [r[0] for r in results]

    # 🔥 RERANK
    final_docs = rerank(query, docs, top_k=2)
    final_docs = list(dict.fromkeys(final_docs))

    return final_docs