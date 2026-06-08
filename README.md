# 🏥 MedIntel AI

> Retrieval-Augmented Generation (RAG) system for medical document question answering using FastAPI, Groq, Supabase pgvector, and LangChain.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.114-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-brightgreen)
![Groq](https://img.shields.io/badge/Groq-Llama3.1-red)

Upload medical PDFs → ask questions in natural language → receive cited, confidence-scored answers in seconds.

---

## 🎯 Features

* Upload and index medical PDFs
* Semantic search using vector embeddings
* Citation-aware answers with source attribution
* Confidence scoring for retrieved evidence
* Supabase pgvector cloud vector database
* Groq-powered LLM responses
* FastAPI backend deployed on Render
* Swagger API documentation

---

## 🧠 What It Does

MedIntel AI enables users to upload medical documents such as:

* Clinical guidelines
* Pharmacology slides
* Medical research papers
* Treatment protocols

Users can then ask questions in natural language.

Every answer includes:

* Retrieved evidence from the uploaded document
* Source document name
* Page number references
* Similarity scores
* Confidence labels (High / Medium / Low)

---

## 🏗️ Architecture

PDF Upload
↓
PyMuPDF Text Extraction
↓
LangChain Text Chunking
↓
all-MiniLM-L6-v2 Embeddings
↓
Supabase PostgreSQL + pgvector
↓
Semantic Similarity Search
↓
Groq Llama 3.1
↓
Answer Generation
↓
Citations + Confidence Scores
↓
FastAPI Response

---

## 🛠️ Tech Stack

| Layer            | Technology                             |
| ---------------- | -------------------------------------- |
| Backend          | FastAPI                                |
| LLM              | Groq (Llama 3.1 Instant)               |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database  | Supabase PostgreSQL + pgvector         |
| RAG Framework    | LangChain                              |
| PDF Processing   | PyMuPDF                                |
| API Server       | Uvicorn                                |
| Deployment       | Render                                 |
| Database Hosting | Supabase                               |

---

## 📡 API Endpoints

| Method | Endpoint       | Description              |
| ------ | -------------- | ------------------------ |
| GET    | /api/v1/health | System status            |
| POST   | /api/v1/upload | Upload and index PDF     |
| POST   | /api/v1/query  | Query uploaded documents |
| GET    | /docs          | Swagger Documentation    |

---

## 🚀 Deployment

Backend deployed on Render.

Vector storage hosted on Supabase PostgreSQL with pgvector.

Example deployment flow:

User Query
↓
Render-hosted FastAPI API
↓
Supabase Vector Search
↓
Groq LLM
↓
Grounded Answer + Citations

---

## 📊 Example Query

Question:

What is MIC?

Response:

MIC stands for Minimum Inhibitory Concentration.

Source:

* antimicrobials.pdf
* Page 10

Confidence:

* Medium (0.55)

---

## 📁 Project Structure

medintel-ai/

├── app/

│ ├── api/

│ ├── core/

│ ├── models/

│ └── main.py

├── requirements.txt

├── README.md

└── .env

---

## 💡 Key Learning Outcomes

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* pgvector
* FastAPI API Development
* Cloud Deployment
* LLM Integration
* Medical Document Processing

---

## 📝 Resume Highlight

Built MedIntel AI, a cloud-deployed Retrieval-Augmented Generation (RAG) platform for medical document Q&A using FastAPI, LangChain, Groq Llama 3.1, and Supabase pgvector. Implemented semantic retrieval, citation-aware responses, confidence scoring, and cloud-hosted vector search over uploaded medical PDFs.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical advice.
