from fastapi import FastAPI
from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_text
from retrieval.vector_store import create_vector_index
from retrieval.retriever import search
from llm.generator import generate_answer

app = FastAPI()

# Load data once at startup
docs = load_documents("data")

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

index, embeddings = create_vector_index(chunks)


@app.get("/")
def home():
    return {"message": "RAG platform running"}


@app.get("/ask")
def ask(question: str):

    # Step 1: Retrieve relevant chunks
    results = search(question, index, chunks)

    # Step 2: Combine context
    context = "\n".join(results)

    # Step 3: Generate answer
    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "sources": results
    }