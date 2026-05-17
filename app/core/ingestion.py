"""
app/core/ingestion.py
PDF → text → chunks → embeddings → ChromaDB
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── 1. PDF → raw text with page metadata ────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF.
    Returns list of {"page": int, "text": str, "source": filename}
    """
    pages = []
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).name

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:  # skip blank pages
            pages.append({
                "page": page_num,
                "text": text,
                "source": filename,
                "total_pages": len(doc),
            })

    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {filename}")
    return pages


# ── 2. Pages → LangChain Documents with rich metadata ───────────────────────

def pages_to_documents(pages: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw page dicts to LangChain Document objects."""
    return [
        Document(
            page_content=p["text"],
            metadata={
                "source": p["source"],
                "page": p["page"],
                "total_pages": p["total_pages"],
            },
        )
        for p in pages
    ]


# ── 3. Documents → overlapping chunks ───────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval.
    Chunk metadata inherits source + page from parent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # Enrich each chunk with a human-readable citation string
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["citation"] = (
            f"{chunk.metadata['source']} — page {chunk.metadata['page']}"
        )

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


# ── 4. High-level pipeline ───────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> List[Document]:
    """
    Full pipeline: PDF file → list of enriched chunk Documents.
    Call this from the API upload endpoint.
    """
    pages = extract_text_from_pdf(pdf_path)
    docs = pages_to_documents(pages)
    chunks = chunk_documents(docs)
    return chunks
