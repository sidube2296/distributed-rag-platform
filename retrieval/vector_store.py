from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_index(chunks):

    # Convert text → embeddings
    embeddings = model.encode(chunks)

    # Get dimension size
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    index.add(np.array(embeddings))

    return index, embeddings