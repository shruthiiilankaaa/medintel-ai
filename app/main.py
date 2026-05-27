"""
app/main.py
FastAPI application factory.
Run with: uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router
from app.api.query import router as query_router
from app.core.config import get_settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan: warm up models on startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the embedding model and LLM at startup to avoid cold starts."""
    logger.info("⚡ MedIntel AI starting up — warming up models...")
    from app.core.vectorstore import get_embeddings
    from app.core.rag_chain import get_llm
    get_embeddings()  # loads sentence-transformer
    logger.info("✅ Using Groq API for LLM — no local model needed.")
    yield
    logger.info("🛑 MedIntel AI shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedIntel AI",
    description=(
        "Retrieval-Augmented Generation (RAG) system for medical document Q&A. "
        "Upload PDFs, ask questions, get cited answers with confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow Streamlit frontend at any port during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "MedIntel AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
