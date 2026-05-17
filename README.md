# 🏥 MedIntel AI

> **Retrieval-Augmented Generation (RAG) system for medical document Q&A**  
> Upload PDFs → ask questions → get cited, confidence-scored answers in seconds.

---

## 🎯 What It Does

MedIntel AI lets you upload medical PDFs (clinical guidelines, research papers, drug handbooks) and query them in natural language. Every answer is:

- **Grounded** — generated only from your uploaded documents, never hallucinated
- **Cited** — each answer links back to the exact source document + page number
- **Scored** — a confidence score (High / Medium / Low) tells you how strongly the evidence supports the answer

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
| **LLM** | HuggingFace `google/flan-t5-base` (swap for Mistral-7B for better quality) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (80 MB, CPU-friendly) |
| **Vector DB** | ChromaDB (persistent, local) |
| **RAG Framework** | LangChain |
| **PDF Parsing** | PyMuPDF (fitz) |
| **API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Containerisation** | Docker + docker-compose |
| **Testing** | pytest |

---

## 🚀 Quick Start

### Option A — Docker (recommended, zero setup)

```bash
git clone https://github.com/yourname/medintel-ai.git
cd medintel-ai
docker-compose up --build
```

- **API**: http://localhost:8000/docs
- **App**: http://localhost:8501

### Option B — Local Python

```bash
# 1. Clone + venv
git clone https://github.com/yourname/medintel-ai.git
cd medintel-ai
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Start FastAPI
uvicorn app.main:app --reload --port 8000

# 4. Start Streamlit (new terminal)
streamlit run frontend/app.py
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

## 🔧 Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `google/flan-t5-base` | HuggingFace generative model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_DOCS` | `4` | Chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | `0.35` | Min cosine similarity to use a chunk |

### Upgrading the LLM

Change in `.env`:
```
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```
Requires ~8 GB RAM. Or use OpenAI by replacing the pipeline in `rag_chain.py`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System status + document count |
| `POST` | `/api/v1/upload` | Upload + index a PDF |
| `POST` | `/api/v1/query` | Ask a question, get RAG answer |
| `GET` | `/docs` | Swagger UI |

### Example query (curl)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the contraindications for ibuprofen?", "top_k": 4}'
```

---

## 📝 Resume Bullets

```
• Built a production-grade RAG healthcare assistant (MedIntel AI) using LangChain,
  HuggingFace Transformers (all-MiniLM-L6-v2, flan-t5), and ChromaDB for semantic
  retrieval over medical PDFs with citation-based, confidence-scored answers.

• Engineered a full-stack AI application with FastAPI (REST backend), Streamlit
  (chat UI), and Docker Compose for containerised deployment, processing PDFs into
  512-token overlapping chunks for high-precision semantic search.

• Implemented cosine similarity–based confidence scoring to surface retrieval
  quality to end users, reducing hallucination risk in medical Q&A workflows.
```

---

## ⚠️ Disclaimer

This tool is for research and educational purposes. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

*Built with LangChain · HuggingFace · ChromaDB · FastAPI · Streamlit · Docker*
