"""
app/api/query.py
POST /query  — answer a medical question using RAG.
GET  /health — system status.
"""
import logging
from fastapi import APIRouter, HTTPException

from app.core.rag_chain import answer_query
from app.core.vectorstore import collection_size
from app.models.schemas import (
    QueryRequest, QueryResponse, CitationOut,
    HealthResponse, confidence_label
)
from app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Ask a question about the uploaded medical documents.

    Returns:
    - **answer**: Context-aware answer grounded in the PDF content
    - **citations**: Source chunks with page numbers and similarity scores
    - **confidence**: Mean cosine similarity of retrieved context (0–1)
    - **confidence_label**: Human-readable: High / Medium / Low
    """
    if collection_size() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first via POST /upload."
        )

    try:
        rag_response = answer_query(request.query)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

    citations_out = [
        CitationOut(
            source=c.source,
            page=c.page,
            chunk_preview=c.chunk_text,
            score=c.score,
        )
        for c in rag_response.citations
    ]

    return QueryResponse(
        answer=rag_response.answer,
        citations=citations_out,
        confidence=rag_response.confidence,
        confidence_label=confidence_label(rag_response.confidence),
        query=rag_response.query,
        total_citations=len(citations_out),
    )

@router.get("/health", tags=["System"])
async def health_check():
    try:
        docs = collection_size()
    except Exception as e:
        docs = -1

    return {
        "status": "ok",
        "documents_indexed": docs,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }