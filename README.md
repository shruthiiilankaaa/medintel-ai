# 🏥 MedIntel AI

> A Retrieval-Augmented Generation (RAG) system for medical document question answering using FastAPI, LangChain, Groq, and Supabase pgvector.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.114-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Supabase](https://img.shields.io/badge/Supabase-pgvector-brightgreen)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1-red)
![Render](https://img.shields.io/badge/Deployment-Render-purple)

---

## 🚀 Overview

MedIntel AI allows users to upload medical PDFs and ask questions in natural language.

Instead of manually searching through lengthy clinical documents, users can ask questions conversationally and receive answers grounded in the uploaded content, complete with source citations, page references, similarity scores, and confidence levels.

This project was built to explore modern Retrieval-Augmented Generation (RAG) architectures and gain hands-on experience with vector databases, semantic search, cloud deployment, and LLM integration.

---

## ✨ Features

### 📄 Medical PDF Processing

* Upload medical PDFs, clinical guidelines, research papers, and lecture slides
* Automatic text extraction using PyMuPDF
* Intelligent document chunking for efficient retrieval

### 🔍 Semantic Search

* Generates embeddings using `all-MiniLM-L6-v2`
* Stores vectors in Supabase PostgreSQL with pgvector
* Retrieves the most relevant chunks using similarity search

### 🤖 Context-Aware Answer Generation

* Uses Groq's Llama 3.1 model for response generation
* Answers are grounded only in retrieved document context
* Reduces hallucinations through retrieval-based prompting

### 📌 Source Attribution

Every response includes:

* Source document name
* Page number
* Similarity score
* Confidence level

### ☁️ Cloud Deployment

* Backend deployed on Render
* Vector database hosted on Supabase
* Public API with Swagger documentation

---

## 🏗️ System Architecture

```text
PDF Upload
    │
    ▼
PyMuPDF Text Extraction
    │
    ▼
LangChain Chunking
    │
    ▼
Sentence Transformer Embeddings
(all-MiniLM-L6-v2)
    │
    ▼
Supabase PostgreSQL + pgvector
    │
    ▼
Semantic Similarity Search
    │
    ▼
Groq Llama 3.1
    │
    ▼
Answer Generation
    │
    ▼
Citations + Confidence Score
    │
    ▼
FastAPI Response
```

---

## 🛠️ Tech Stack

| Layer            | Technology                             |
| ---------------- | -------------------------------------- |
| Backend API      | FastAPI                                |
| API Server       | Uvicorn                                |
| RAG Framework    | LangChain                              |
| LLM              | Groq (Llama 3.1 Instant)               |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database  | Supabase PostgreSQL + pgvector         |
| PDF Processing   | PyMuPDF                                |
| Configuration    | Pydantic Settings                      |
| Deployment       | Render                                 |
| Database Hosting | Supabase                               |

---

## 📊 Example Workflow

### User Query

```text
What is MIC?
```

### Retrieval Pipeline

1. Convert query into embeddings
2. Search pgvector for semantically similar chunks
3. Retrieve top matching document sections
4. Build contextual prompt
5. Send context to Groq Llama 3.1
6. Generate grounded answer
7. Return citations and confidence score

### Example Response

```json
{
  "answer": "MIC stands for Minimum Inhibitory Concentration.",
  "confidence": 0.55,
  "confidence_label": "Medium",
  "citations": [
    {
      "source": "antimicrobials.pdf",
      "page": 10,
      "score": 0.594
    }
  ]
}
```

---

## 📡 API Endpoints

| Method | Endpoint         | Description                               |
| ------ | ---------------- | ----------------------------------------- |
| GET    | `/api/v1/health` | Service health and indexed document count |
| POST   | `/api/v1/upload` | Upload and index a PDF                    |
| POST   | `/api/v1/query`  | Query uploaded documents                  |
| GET    | `/docs`          | Interactive Swagger documentation         |

---

## ⚙️ Environment Variables

Create a `.env` file:

```env
# Groq
GROQ_API_KEY=your_groq_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_DB_URL=your_postgresql_connection_string
```

---

## 💻 Local Setup

### Clone Repository

```bash
git clone https://github.com/yourusername/medintel-ai.git

cd medintel-ai
```

### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

## 📚 Key Concepts Learned

### Retrieval-Augmented Generation (RAG)

Built a complete RAG pipeline that retrieves relevant document context before generating answers.

### Semantic Search

Implemented vector similarity search using embeddings instead of traditional keyword matching.

### Vector Databases

Worked with PostgreSQL + pgvector to store and retrieve high-dimensional embeddings.

### Cloud Deployment

Deployed and debugged production workloads on Render and Supabase.

### LLM Integration

Integrated Groq-hosted Llama 3.1 models for grounded answer generation.

---

## 🔧 Engineering Challenges Solved

During development, several production-level issues were encountered and resolved:

* Render deployment failures and startup issues
* Memory constraints while loading embedding models
* Supabase PostgreSQL connectivity problems
* pgvector integration and similarity search debugging
* Retrieval result formatting issues
* Groq model deprecation and migration
* Environment variable management in production

These challenges provided practical experience in debugging cloud-native AI systems.

---

## 🚀 Future Improvements

* Multi-document retrieval
* Hybrid search (BM25 + Vector Search)
* Cross-encoder reranking
* User authentication
* Document-level access control
* Frontend dashboard for document management

---

## 📝 Resume Highlight

Built and deployed a Retrieval-Augmented Generation (RAG) platform for medical document question answering using FastAPI, LangChain, Groq Llama 3.1, and Supabase pgvector. Implemented semantic search, citation-aware responses, confidence scoring, and cloud-hosted vector retrieval over uploaded medical PDFs.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical advice.
