from fastapi import FastAPI
from retrieval.db_search import db_search
from llm.generator import generate_answer
import time

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG platform running with PostgreSQL 🚀"}


@app.get("/ask")
def ask(question: str):

    start = time.time()

    # Step 1: Retrieve from DB
    results = db_search(question, top_k=5)

    # Step 2: Combine context
    context = "\n".join(results)

    # Step 3: Generate answer
    answer = generate_answer(context, question)

    end = time.time()

    return {
        "question": question,
        "answer": answer.strip(),
        "sources": [r[:200] + "..." for r in results],
        "latency": round(end - start, 2)
    }