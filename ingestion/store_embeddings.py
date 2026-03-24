from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_text
from sentence_transformers import SentenceTransformer
import psycopg2

# DB connection
conn = psycopg2.connect(
    dbname="rag_db",
    user="siddhi2218",   # ✅ your username
    password="",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
docs = load_documents("data/arxiv")

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

print(f"Total chunks: {len(chunks)}")

# Insert into DB
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()

    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (chunk, embedding)
    )

    if i % 100 == 0:
        print(f"Inserted {i} chunks...")

conn.commit()

print("✅ All embeddings stored in DB")
