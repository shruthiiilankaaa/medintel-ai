"""
app/core/rag_chain.py
RAG pipeline using Groq API for LLM (no local model = no OOM on Render)
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List

from langchain.schema import Document
from app.core.vectorstore import similarity_search_with_scores
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


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


SYSTEM_PROMPT = """You are MedIntel AI, a precise medical document assistant.
Answer the question using ONLY the provided context from medical documents.
If the context does not contain enough information, say "I cannot find a confident answer in the uploaded documents."
Always be factual and concise."""


def build_prompt(query: str, context_chunks: List[Document]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('citation', 'unknown')}]\n{doc.page_content}"
        for doc in context_chunks
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT FROM MEDICAL DOCUMENTS:\n{context_text}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


def compute_confidence(scores: List[float]) -> float:
    if not scores:
        return 0.0
    top_scores = sorted(scores, reverse=True)[:2]
    return round(sum(top_scores) / len(top_scores), 3)


def generate_answer(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def answer_query(query: str) -> RAGResponse:
    results = similarity_search_with_scores(query, k=settings.top_k_docs)

    if not results:
        return RAGResponse(
            answer="No documents uploaded yet. Please upload a medical PDF first.",
            citations=[],
            confidence=0.0,
            query=query,
        )

    citations = []
    strong_docs = []
    scores = []

    for doc, score in results:
        citations.append(Citation(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", 0),
            chunk_text=doc.page_content[:300],
            score=round(score, 3),
        ))
        if score >= settings.confidence_threshold:
            strong_docs.append(doc)
            scores.append(score)

    context_docs = strong_docs if strong_docs else [r[0] for r in results[:2]]
    prompt = build_prompt(query, context_docs)

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        logger.error(f"Groq API failed: {e}")
        answer = "Error generating answer. Check your GROQ_API_KEY."

    return RAGResponse(
        answer=answer,
        citations=citations,
        confidence=compute_confidence(scores if scores else [r[1] for r in results]),
        query=query,
    )