"""
app/core/vectorstore.py
HuggingFace embeddings + Supabase pgvector vector store management.
Handles: embeddings, storing documents, semantic search.
"""

import logging
import json
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from sqlalchemy import create_engine, text

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Database Engine ───────────────────────────────────────────────────────────

engine = create_engine(settings.supabase_db_url)

# ── Singleton embedding model ────────────────────────────────────────────────

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer model once and cache it.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Embedding model loaded.")

    return _embedding_model


# ── Operations ───────────────────────────────────────────────────────────────

def add_documents(chunks: List[Document]) -> int:
    """
    Embed and store chunks in Supabase pgvector.
    """
    embeddings_model = get_embeddings()

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)

    with engine.begin() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            conn.execute(
                text("""
                    INSERT INTO documents (content, metadata, embedding)
                    VALUES (:content, :metadata, :embedding)
                """),
                {
                    "content": chunk.page_content,
                    "metadata": json.dumps(chunk.metadata),
                    "embedding": str(embedding),
                },
            )

    logger.info(f"Added {len(chunks)} chunks to Supabase.")
    return len(chunks)


def similarity_search_with_scores(
    query: str,
    k: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Return top-k relevant chunks as (Document, similarity_score).
    """
    k = k or settings.top_k_docs

    embeddings_model = get_embeddings()
    query_embedding = embeddings_model.embed_query(query)

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT content, metadata, similarity
                FROM match_documents(
                    CAST(:query_embedding AS vector),
                    :match_count
                )
            """),
            {
                "query_embedding": str(query_embedding),
                "match_count": k,
            },
        )

        rows = result.fetchall()

    results = []

    for content, metadata, similarity in rows:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        doc = Document(
            page_content=content,
            metadata=metadata or {},
        )

        results.append(
            (
                doc,
                float(similarity),
            )
        )

    logger.info(f"Retrieved {len(results)} similar documents.")

    return results


def collection_size() -> int:
    """
    Return number of stored chunks.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM documents")
        )

        count = result.scalar()

    return count


def clear_collection() -> None:
    """
    Delete all documents from Supabase documents table.
    """
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM documents")
        )

    logger.warning("Supabase vector store cleared.")