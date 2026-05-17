"""
app/api/upload.py
POST /upload  — accept a PDF, ingest it, embed it, store in ChromaDB.
"""
import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.ingestion import ingest_pdf
from app.core.vectorstore import add_documents, collection_size
from app.models.schemas import UploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"application/pdf", "application/octet-stream"}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a medical PDF for indexing.

    - Extracts text page-by-page using PyMuPDF
    - Splits into overlapping chunks (512 tokens, 64 overlap)
    - Embeds with all-MiniLM-L6-v2
    - Stores in persistent ChromaDB collection
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Read + size check
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB."
        )

    # Write to temp file (PyMuPDF needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest_pdf(tmp_path)
        # Override source with original filename
        for chunk in chunks:
            chunk.metadata["source"] = file.filename
            chunk.metadata["citation"] = (
                f"{file.filename} — page {chunk.metadata['page']}"
            )

        chunks_added = add_documents(chunks)
        total_indexed = collection_size()

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        filename=file.filename,
        chunks_indexed=chunks_added,
        total_documents_indexed=total_indexed,
        message=f"Successfully indexed '{file.filename}' ({chunks_added} chunks).",
    )
