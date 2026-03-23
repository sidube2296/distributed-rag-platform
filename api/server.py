from fastapi import FastAPI
from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_text
from retrieval.vector_store import create_vector_index
from retrieval.retriever import search
from llm.generator import generate_answer
from retrieval.bm25_retriever import create_bm25_index
from retrieval.retriever import hybrid_search

app = FastAPI()

# Load data once at startup
docs = load_documents("data")

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))


bm25, tokenized_chunks = create_bm25_index(chunks)
index, embeddings = create_vector_index(chunks)


@app.get("/")
def home():
    return {"message": "RAG platform running"}


@app.get("/ask")
def ask(question: str):

    # Step 1: Retrieve relevant chunks
    results = hybrid_search(question, index, chunks, bm25, tokenized_chunks)

    # Step 2: Combine context
    context = "\n".join(results)

    # Step 3: Generate answer
    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "sources": results
    }