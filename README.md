# 🏥 MedIntel AI
> Retrieval-Augmented Generation (RAG) system for medical document Q&A

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.114-green) ![LangChain](https://img.shields.io/badge/LangChain-0.2-orange) ![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

Upload medical PDFs → ask questions in natural language → get cited, confidence-scored answers in seconds.
'NEED TO DO THE DOCKER PART'
---

## 🎯 What It Does

MedIntel AI lets you upload medical PDFs (clinical guidelines, research papers, drug handbooks, pharmacology slides) and query them conversationally. Every answer is:

- **Grounded** — generated only from your uploaded documents, never hallucinated
- **Cited** — each answer links back to the exact source document + page number
- **Scored** — a confidence score (High / Medium / Low) tells you how strongly the evidence supports the answer

---

## 🖥️ Demo

> Upload a medical PDF → ask a clinical question → get a cited answer with confidence score

**Example query:** *"Which antibiotics require therapeutic drug monitoring?"*

**Answer:** Retrieved from page 13 of the uploaded pharmacology document, citing Vancomycin and Aminoglycosides with similarity scores and page references.

---

## 🧱 Architecture

```
PDF Upload
    │
    ▼
PyMuPDF (text extraction, page-by-page)
    │
    ▼
LangChain RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
    │
    ▼
HuggingFace all-MiniLM-L6-v2 (sentence embeddings, 384-dim)
    │
    ▼
ChromaDB (persistent vector store)
    │
    ▼  ← User Query (also embedded)
Semantic Similarity Search (cosine, top-k=4)
    │
    ▼
Prompt Builder (system prompt + retrieved context + question)
    │
    ▼
google/flan-t5-base (LLM answer generation)
    │
    ▼
FastAPI Response (answer + citations + confidence score)
    │
    ▼
Streamlit UI (chat interface + citation cards)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | HuggingFace `google/flan-t5-base` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent, local) |
| RAG Framework | LangChain |
| PDF Parsing | PyMuPDF (fitz) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Testing | pytest (10/10 passing) |
| Containerisation | Docker + docker-compose |

---

## 🚀 Quick Start

### Option A — Local Python

```bash
# 1. Clone and set up
git clone https://github.com/YOURNAME/medintel-ai.git
cd medintel-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Start the API
uvicorn app.main:app --reload --port 8000

# 3. Start the UI (new terminal)
streamlit run frontend/app.py
```

- API docs: http://localhost:8000/docs
- Chat UI: http://localhost:8501

### Option B — Docker

```bash
docker-compose up --build
```

---

## 📁 Project Structure
```
medintel-ai/
├── app/
│   ├── main.py                # FastAPI app factory + lifespan
│   ├── api/
│   │   ├── upload.py          # POST /api/v1/upload
│   │   └── query.py           # POST /api/v1/query  GET /api/v1/health
│   ├── core/
│   │   ├── config.py          # Pydantic settings from .env
│   │   ├── ingestion.py       # PDF → chunks pipeline
│   │   ├── vectorstore.py     # ChromaDB + HuggingFace embeddings
│   │   └── rag_chain.py       # RAG pipeline + confidence scoring
│   └── models/
│       └── schemas.py         # Pydantic request/response models
├── frontend/
│   └── app.py                 # Streamlit chat UI
├── tests/
│   └── test_pipeline.py       # pytest unit + integration tests
├── data/
│   └── chroma_db/             # Persistent vector store (git-ignored)
├── .env                       # Environment config
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.frontend
└── docker-compose.yml
```
---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System status + document count |
| `POST` | `/api/v1/upload` | Upload + index a PDF |
| `POST` | `/api/v1/query` | Ask a question, get RAG answer |
| `GET` | `/docs` | Swagger UI |

---

## ⚙️ Configuration

Edit `.env` to customise:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `google/flan-t5-base` | Generative model (swap for Mistral-7B for better answers) |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `TOP_K_DOCS` | `4` | Chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | `0.35` | Min cosine similarity to use a chunk |

---

## 🧪 Tests

```bash
pytest tests/ -v
# 10 passed, 4 warnings
```

---

## 📝 Resume Bullet

> Built a production-grade Retrieval-Augmented Generation (RAG) healthcare assistant (MedIntel AI) using LangChain, HuggingFace Transformers (all-MiniLM-L6-v2, flan-t5), and ChromaDB for semantic retrieval over medical PDFs, with citation-based answers, cosine similarity confidence scoring, a FastAPI REST backend, and a Streamlit chat interface — containerised with Docker.

---

## ⚠️ Disclaimer

This tool is for research and educational purposes only. Not a substitute for professional medical advice.

---

*Built with LangChain · HuggingFace · ChromaDB · FastAPI · Streamlit · Docker*
