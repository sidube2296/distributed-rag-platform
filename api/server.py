from fastapi import FastAPI
from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_text
from retrieval.vector_store import create_vector_index
from llm.generator import generate_answer
from retrieval.bm25_retriever import create_bm25_index
from retrieval.retriever import hybrid_search
from retrieval.reranker import rerank

app = FastAPI()

# ---------------------------
# Load and preprocess data (runs once at startup)
# ---------------------------

docs = load_documents("data")

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

# Create BM25 index
bm25, tokenized_chunks = create_bm25_index(chunks)

# Create vector index
index, embeddings = create_vector_index(chunks)


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def home():
    return {"message": "RAG platform running"}


@app.get("/ask")
def ask(question: str):

    # Basic validation
    if not question:
        return {"error": "Question cannot be empty"}

    try:
        # Step 1: Hybrid retrieval
        initial_results = hybrid_search(
            query=question,
            index=index,
            chunks=chunks,
            bm25=bm25,
            tokenized_chunks=tokenized_chunks,
            top_k=2
        )
	
	# Step 2: Rerank them
        results = rerank(question, initial_results, top_k=2)

        # Step 2: Build context
        context = "\n".join(results)

        # Step 3: Generate answer
        answer = generate_answer(context, question)

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": results
        }

    except Exception as e:
        return {"error": str(e)}