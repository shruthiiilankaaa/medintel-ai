"""
app/models/schemas.py
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class CitationOut(BaseModel):
    source: str
    page: int
    chunk_preview: str = Field(..., description="First 300 chars of the matched chunk")
    score: float = Field(..., description="Cosine similarity score")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                       example="What are the contraindications for aspirin?")
    top_k: Optional[int] = Field(default=4, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationOut]
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label: str  # "High" / "Medium" / "Low"
    query: str
    total_citations: int


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    total_documents_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    embedding_model: str
    llm_model: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def confidence_label(score: float) -> str:
    if score >= 0.65:
        return "High"
    elif score >= 0.40:
        return "Medium"
    return "Low"
