# 📚 Hybrid RAG (Retrieval-Augmented Generation) using FastAPI, Qdrant & Gemini

A production-style **Hybrid Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents, indexes them using **Dense + Sparse Embeddings**, stores them in **Qdrant Vector Database**, and answers user questions using **Google Gemini 2.5 Flash**.

---

# 🚀 Features

- 📄 Upload PDF documents
- 📖 Automatic PDF text extraction
- ✂️ Intelligent document chunking
- 🧠 Dense Embeddings using **BAAI/bge-small-en-v1.5**
- 🔍 Sparse Embeddings using **SPLADE**
- 🗄 Hybrid Vector Search using **Qdrant**
- ⚡ FastAPI REST APIs
- 📘 Interactive Swagger UI
- 🤖 Answer Generation using **Gemini 2.5 Flash**
- 🔀 Hybrid Retrieval using **Reciprocal Rank Fusion (RRF)**

---

# 🏗 Architecture

```
                +--------------------+
                |     Upload PDF     |
                +---------+----------+
                          |
                          v
                +--------------------+
                |  PDF Text Extract  |
                +---------+----------+
                          |
                          v
                +--------------------+
                |    Text Chunking   |
                +---------+----------+
                          |
          +---------------+---------------+
          |                               |
          v                               v
+---------------------+      +----------------------+
| Dense Embedding     |      | Sparse Embedding     |
| BGE-small-en-v1.5   |      | SPLADE               |
+----------+----------+      +----------+-----------+
           |                            |
           +-------------+--------------+
                         |
                         v
                +----------------------+
                |   Qdrant Vector DB   |
                | Dense + Sparse Index |
                +----------+-----------+
                           |
                    User Question
                           |
                           v
              Dense + Sparse Retrieval
                           |
                           v
           Reciprocal Rank Fusion (RRF)
                           |
                           v
                 Relevant Context
                           |
                           v
                 Gemini 2.5 Flash
                           |
                           v
                    Final Response
```

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| FastAPI | REST API Framework |
| Swagger | API Documentation |
| Qdrant | Vector Database |
| Docker | Run Qdrant Container |
| Sentence Transformers | Dense Embeddings |
| SPLADE | Sparse Embeddings |
| Google Gemini | LLM |
| PyMuPDF | PDF Parsing |

---

# 📂 Project Structure

```
RAG_PROJECT/

│

├── api/

│   ├── models.py

│   ├── routes.py

│

├── app/

│   ├── chunker.py

│   ├── config.py

│   ├── embedder.py

│   ├── sparse_embedder.py

│   ├── llm.py

│   ├── pdf_parser.py

│   ├── retriever.py

│   ├── vector_db.py

│

├── data/

│   ├── pdfs/

│   ├── chunks/

│

├── main.py

├── requirements.txt

├── README.md

```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/rag-project.git

cd rag-project
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

# 🐳 Run Qdrant

```bash
docker run -d \
-p 6333:6333 \
-v qdrant_storage:/qdrant/storage \
qdrant/qdrant
```

Verify

```
http://localhost:6333/dashboard
```

---

# ▶ Run FastAPI

```bash
uvicorn main:app --reload
```

Application

```
http://localhost:8000
```

Swagger UI

```
http://localhost:8000/docs
```

---

# 📤 Upload API

```
POST /upload
```

Upload any PDF document.

Example Response

```json
{
    "message":"Document indexed successfully",
    "filename":"sample.pdf",
    "total_chunks":18
}
```

---

# 💬 Chat API

```
POST /chat
```

Request

```json
{
    "question":"What is Retrieval Augmented Generation?"
}
```

Response

```json
{
    "answer":"Retrieval-Augmented Generation (RAG) combines..."
}
```

---

# 🔄 Complete Workflow

### 1. Upload PDF

↓

### 2. Extract Text

↓

### 3. Chunk Document

↓

### 4. Generate Dense Embeddings

Model Used

```
BAAI/bge-small-en-v1.5
```

Dimension

```
384
```

↓

### 5. Generate Sparse Embeddings

Model Used

```
SPLADE
```

↓

### 6. Store in Qdrant

Each chunk stores

```
Dense Vector

Sparse Vector

Page Number

Chunk Text

Source Filename
```

↓

### 7. User Asks Question

↓

### 8. Dense Retrieval

↓

### 9. Sparse Retrieval

↓

### 10. Reciprocal Rank Fusion (RRF)

↓

### 11. Top-K Relevant Chunks

↓

### 12. Gemini Generates Final Answer

---

# 🧠 Embedding Models

## Dense Embedding

Model

```
BAAI/bge-small-en-v1.5
```

Dimension

```
384
```

Purpose

- Semantic Search
- Context Understanding
- Similar Meaning Retrieval

---

## Sparse Embedding

Model

```
SPLADE
```

Purpose

- Exact Keyword Matching
- BM25-like Retrieval
- Better Technical Search

---

# 🗄 Vector Database

Database Used

```
Qdrant
```

Features

- Dense Vector Search
- Sparse Vector Search
- Hybrid Retrieval
- Payload Storage
- HNSW Index
- Cosine Similarity

---

# 🔍 Retrieval Strategy

Hybrid Search

```
Dense Search

+

Sparse Search

↓

Reciprocal Rank Fusion (RRF)

↓

Top-K Documents
```

---

# 🤖 LLM

Model

```
Gemini 2.5 Flash
```

Used For

- Context Aware Question Answering
- Natural Language Generation

---

# 📈 Future Improvements

- Multi PDF Support
- Metadata Filtering
- User Authentication
- Streaming Responses
- Citation Generation
- Reranking Models
- Conversation Memory
- Docker Compose
- Kubernetes Deployment
- Azure/OpenAI Embeddings
- Production Logging
- Monitoring with Prometheus

---

# 📸 Screenshots

## Swagger UI

(Add Screenshot)

---

## Upload API

(Add Screenshot)

---

## Chat API

(Add Screenshot)

---

## Qdrant Dashboard

(Add Screenshot)

---

# 📚 Concepts Covered

- Retrieval Augmented Generation (RAG)
- Dense Embeddings
- Sparse Embeddings
- Hybrid Search
- Vector Databases
- Approximate Nearest Neighbor Search
- HNSW
- Cosine Similarity
- Reciprocal Rank Fusion
- FastAPI
- Swagger/OpenAPI
- Docker
- PDF Parsing
- Prompt Engineering

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Production-ready RAG Pipeline
- Hybrid Retrieval
- FastAPI Backend Development
- Vector Database Integration
- LLM Application Development
- Semantic Search
- Information Retrieval

---

# 📄 License

MIT License

---

# 👨‍💻 Author

Developed by **Your Name**

If you found this project useful, consider giving it a ⭐ on GitHub!
