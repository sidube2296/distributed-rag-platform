# 🚀 Distributed RAG Knowledge Intelligence Platform

An end-to-end **Retrieval Augmented Generation (RAG)** system that enables users to query large-scale document collections and receive accurate, context-aware responses using LLMs.

---

## 📌 Overview

This project implements a production-style **AI knowledge retrieval system** that combines:

* Semantic search (vector embeddings)
* Keyword-based retrieval (BM25)
* Large Language Models (LLMs)
* FastAPI backend for real-time inference

---

## 🧠 Architecture

User Query → Retrieval → Reranking → LLM → Answer + Sources

---

## ⚙️ Tech Stack

* Python
* FastAPI
* FAISS (Vector Database)
* LangChain
* Ollama (Llama 3)
* Sentence Transformers

---

## 🔥 Features

* Hybrid retrieval (vector + keyword search)
* Context-aware LLM responses
* Source citation support
* Modular architecture (ingestion, retrieval, generation)
* Scalable backend using FastAPI

---

## 📂 Project Structure

distributed-rag-platform/
│
├── ingestion/
├── retrieval/
├── llm/
├── api/
├── data/
├── tests/
├── README.md

---

## 🚀 Getting Started

### 1. Clone the repo

git clone https://github.com/sidube2296/distributed-rag-platform.git

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run Ollama

ollama serve

### 4. Start API

uvicorn api.server:app --reload

---

## 📊 Example Query

GET /ask?question=What is transformer?

---

## 📈 Future Improvements

* Reranking with cross-encoders
* Redis caching for performance
* Evaluation pipeline (RAGAS)
* Frontend UI (Streamlit/React)

---

## 💡 Why This Project Matters

This project demonstrates:

* LLM application development
* Scalable backend engineering
* Information retrieval systems
* Real-world AI system design

---

## 👨‍💻 Author

Siddhi Dube

