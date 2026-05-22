"""
app/core/rag_chain.py
The heart of MedIntel AI:
  - Retrieves relevant chunks from Supabase pgvector
  - Builds a context-aware prompt
  - Generates answers via a HuggingFace LLM
  - Returns answer + citations + confidence score
"""

import json
import logging
from dataclasses import dataclass
from typing import List

from transformers import pipeline, Pipeline

from app.core.vectorstore import similarity_search_with_scores
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Citation:
    source: str
    page: int
    chunk_text: str
    score: float

    @property
    def label(self) -> str:
        return f"{self.source} (page {self.page})"


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    confidence: float
    query: str
    context_used: str = ""


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm_pipeline: Pipeline | None = None


def get_llm() -> Pipeline:
    """
    Load HuggingFace generation pipeline once.
    """
    global _llm_pipeline

    if _llm_pipeline is None:
        logger.info(f"Loading LLM: {settings.llm_model}")

        _llm_pipeline = pipeline(
            "text2text-generation",
            model=settings.llm_model,
            max_new_tokens=512,
            do_sample=False,
        )

        logger.info("LLM ready.")

    return _llm_pipeline


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedIntel AI, a precise medical document assistant.

Answer the question using ONLY the provided context from medical documents.

If the context does not contain enough information, say:
"I cannot find a confident answer in the uploaded documents."

Always be factual, concise, and grounded in the context.
Do NOT invent medical information.
"""


def build_prompt(query: str, context_chunks: List[dict]) -> str:
    """
    Build prompt from retrieved Supabase rows.
    """

    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['citation']}]\n{chunk['content']}"
        for chunk in context_chunks
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT FROM MEDICAL DOCUMENTS:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )


# ── Confidence scoring ────────────────────────────────────────────────────────

def compute_confidence(scores: List[float]) -> float:
    if not scores:
        return 0.0

    top_scores = sorted(scores, reverse=True)[:2]

    return round(sum(top_scores) / len(top_scores), 3)


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

def answer_query(query: str) -> RAGResponse:
    """
    Full RAG pipeline using Supabase pgvector.
    """

    # Step 1: Retrieve similar chunks
    results = similarity_search_with_scores(
        query,
        k=settings.top_k_docs,
    )

    if not results:
        return RAGResponse(
            answer="No relevant documents found.",
            citations=[],
            confidence=0.0,
            query=query,
        )

    citations: List[Citation] = []
    context_chunks = []
    scores: List[float] = []

    # Step 2: Process Supabase rows
    for content, metadata, similarity in results:

        # metadata may arrive as string/json
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        citation = Citation(
            source=metadata.get("source", "unknown"),
            page=metadata.get("page", 0),
            chunk_text=content[:300],
            score=round(similarity, 3),
        )

        citations.append(citation)

        if similarity >= settings.confidence_threshold:
            context_chunks.append({
                "content": content,
                "citation": metadata.get(
                    "citation",
                    f"{citation.source} — page {citation.page}"
                ),
            })

            scores.append(similarity)

    # fallback if no strong matches
    if not context_chunks:
        for content, metadata, similarity in results[:2]:

            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            context_chunks.append({
                "content": content,
                "citation": metadata.get(
                    "citation",
                    f"{metadata.get('source', 'unknown')} — page {metadata.get('page', 0)}"
                ),
            })

    # Step 3: Build prompt
    prompt = build_prompt(query, context_chunks)

    # Step 4: Generate answer
    llm = get_llm()

    try:
        output = llm(prompt)
        answer = output[0]["generated_text"].strip()

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")

        answer = "Error generating answer. Please try again."

    # Step 5: Confidence
    confidence = compute_confidence(
        scores if scores else [r[2] for r in results]
    )

    return RAGResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        query=query,
        context_used=prompt[:500] + "...",
    )