"""
app/core/vectorstore.py
HuggingFace embeddings + ChromaDB vector store management.
Handles: initialisation, adding documents, semantic search.
"""
import logging
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Singleton embedding model (expensive to load repeatedly) ─────────────────

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer model once and cache it.
    all-MiniLM-L6-v2 is 80 MB, fast on CPU, great for medical text retrieval.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        )
        logger.info("Embedding model loaded.")
    return _embedding_model


# ── Singleton vector store ────────────────────────────────────────────────────

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """
    Return a persistent ChromaDB instance.
    Creates the collection if it doesn't exist yet.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(
            f"ChromaDB initialised at {settings.chroma_persist_dir}"
            f" | collection: {settings.chroma_collection}"
        )
    return _vectorstore


# ── Operations ────────────────────────────────────────────────────────────────

def add_documents(chunks: List[Document]) -> int:
    """
    Embed and store chunks in ChromaDB.
    Returns number of chunks added.
    """
    vs = get_vectorstore()
    vs.add_documents(chunks)
    vs.persist()
    logger.info(f"Added {len(chunks)} chunks to vector store.")
    return len(chunks)


def similarity_search_with_scores(
    query: str,
    k: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Return the top-k most relevant chunks with their cosine similarity scores.
    Score range: 0.0 (unrelated) – 1.0 (identical).
    """
    k = k or settings.top_k_docs
    vs = get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    return results  # [(Document, score), ...]


def collection_size() -> int:
    """How many chunks are currently indexed."""
    vs = get_vectorstore()
    return vs._collection.count()


def clear_collection() -> None:
    """Delete all documents from the collection (useful for testing)."""
    global _vectorstore
    vs = get_vectorstore()
    vs.delete_collection()
    _vectorstore = None
    logger.warning("Vector store collection cleared.")
