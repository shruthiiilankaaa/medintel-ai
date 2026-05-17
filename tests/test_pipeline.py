"""
tests/test_pipeline.py
Unit + integration tests for the RAG pipeline.
Run with: pytest tests/ -v
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

# ── Ingestion tests ───────────────────────────────────────────────────────────

def test_chunk_documents():
    """Chunker should produce overlapping chunks with metadata."""
    from langchain.schema import Document
    from app.core.ingestion import chunk_documents

    docs = [Document(
        page_content="A " * 300,  # 600 chars, should split
        metadata={"source": "test.pdf", "page": 1, "total_pages": 1}
    )]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "citation" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) <= 600  # within chunk size


def test_chunk_preserves_metadata():
    """Each chunk should carry source and page from parent."""
    from langchain.schema import Document
    from app.core.ingestion import chunk_documents

    docs = [Document(
        page_content="Medical text " * 50,
        metadata={"source": "drug_guide.pdf", "page": 3, "total_pages": 10}
    )]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "drug_guide.pdf"
        assert chunk.metadata["page"] == 3


# ── Confidence scoring tests ───────────────────────────────────────────────────

def test_confidence_high():
    from app.core.rag_chain import compute_confidence
    assert compute_confidence([0.9, 0.85, 0.7]) >= 0.8


def test_confidence_low():
    from app.core.rag_chain import compute_confidence
    assert compute_confidence([0.2, 0.15]) < 0.3


def test_confidence_empty():
    from app.core.rag_chain import compute_confidence
    assert compute_confidence([]) == 0.0


# ── Schema / label tests ──────────────────────────────────────────────────────

def test_confidence_labels():
    from app.models.schemas import confidence_label
    assert confidence_label(0.8) == "High"
    assert confidence_label(0.5) == "Medium"
    assert confidence_label(0.2) == "Low"
    assert confidence_label(0.65) == "High"
    assert confidence_label(0.40) == "Medium"


# ── Prompt builder test ───────────────────────────────────────────────────────

def test_build_prompt_contains_query():
    from langchain.schema import Document
    from app.core.rag_chain import build_prompt

    docs = [Document(
        page_content="Aspirin inhibits COX-1 and COX-2 enzymes.",
        metadata={"citation": "drug_handbook.pdf — page 12"}
    )]
    prompt = build_prompt("What does aspirin inhibit?", docs)
    assert "aspirin" in prompt.lower()
    assert "COX-1" in prompt
    assert "drug_handbook.pdf" in prompt


# ── FastAPI endpoint tests ────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "documents_indexed" in data


def test_query_no_docs_returns_400(client):
    """Query with empty index should return 400."""
    with patch("app.api.query.collection_size", return_value=0):
        resp = client.post("/api/v1/query", json={"query": "What is the dosage?"})
        assert resp.status_code == 400


def test_upload_non_pdf_rejected(client):
    resp = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", b"some text", "text/plain")},
    )
    assert resp.status_code == 400
