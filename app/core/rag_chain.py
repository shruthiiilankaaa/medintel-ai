"""
app/core/rag_chain.py
The heart of MedIntel AI:
  - Retrieves relevant chunks from ChromaDB
  - Builds a context-aware prompt
  - Generates answers via a HuggingFace LLM (flan-t5 by default)
  - Returns answer + citations + confidence score
"""
import logging
from dataclasses import dataclass, field
from typing import List

from transformers import pipeline, Pipeline
from langchain.schema import Document

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
    confidence: float          # 0.0 – 1.0 (mean retrieval score of top docs)
    query: str
    context_used: str = ""


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm_pipeline: Pipeline | None = None


def get_llm() -> Pipeline:
    """
    Load a HuggingFace seq2seq pipeline once.
    flan-t5-base works on CPU, no GPU required.
    Swap to a larger model in .env for better answers.
    """
    global _llm_pipeline
    if _llm_pipeline is None:
        logger.info(f"Loading LLM: {settings.llm_model}")
        _llm_pipeline = pipeline(
            "text2text-generation",
            model=settings.llm_model,
            max_new_tokens=512,
            do_sample=False,   # deterministic for medical Q&A
        )
        logger.info("LLM ready.")
    return _llm_pipeline


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedIntel AI, a precise medical document assistant.
Answer the question using ONLY the provided context from medical documents.
If the context does not contain enough information, say "I cannot find a confident answer in the uploaded documents."
Always be factual, concise, and cite specific details from the context.
Do NOT invent medical information beyond what is in the context."""


def build_prompt(query: str, context_chunks: List[Document]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('citation', 'unknown')}]\n{doc.page_content}"
        for doc in context_chunks
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT FROM MEDICAL DOCUMENTS:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )


# ── Confidence scoring ────────────────────────────────────────────────────────

def compute_confidence(scores: List[float]) -> float:
    """
    Confidence = mean of top retrieval scores.
    Scores are cosine similarities [0, 1] after normalization.
    We cap at the top-2 to avoid dilution from weak matches.
    """
    if not scores:
        return 0.0
    top_scores = sorted(scores, reverse=True)[:2]
    return round(sum(top_scores) / len(top_scores), 3)


# ── Main RAG function ─────────────────────────────────────────────────────────

def answer_query(query: str) -> RAGResponse:
    """
    Full RAG pipeline:
      1. Embed query → retrieve top-k chunks from ChromaDB
      2. Filter by confidence threshold
      3. Build prompt with retrieved context
      4. Generate answer via LLM
      5. Return structured response with citations + confidence
    """
    # Step 1: Retrieve
    results = similarity_search_with_scores(query, k=settings.top_k_docs)

    if not results:
        return RAGResponse(
            answer="No documents have been uploaded yet. Please upload a medical PDF first.",
            citations=[],
            confidence=0.0,
            query=query,
        )

    # Step 2: Build citations and filter weak results
    citations: List[Citation] = []
    strong_docs: List[Document] = []
    scores: List[float] = []

    for doc, score in results:
        citations.append(Citation(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", 0),
            chunk_text=doc.page_content[:300],  # preview
            score=round(score, 3),
        ))
        if score >= settings.confidence_threshold:
            strong_docs.append(doc)
            scores.append(score)

    # Use strong docs if available, else fall back to all retrieved
    context_docs = strong_docs if strong_docs else [r[0] for r in results[:2]]

    # Step 3 & 4: Prompt → LLM
    prompt = build_prompt(query, context_docs)
    llm = get_llm()

    try:
        output = llm(prompt)
        answer = output[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "Error generating answer. Please try again."

    # Step 5: Assemble response
    confidence = compute_confidence(scores if scores else [r[1] for r in results])

    return RAGResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        query=query,
        context_used=prompt[:500] + "...",  # for debugging
    )
