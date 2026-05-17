"""
frontend/app.py
Streamlit UI for MedIntel AI
Run with: streamlit run frontend/app.py
"""
import streamlit as st
import requests
import json
import os
from datetime import datetime

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedIntel AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subheader { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .confidence-high   { color: #2e7d32; font-weight: 600; }
    .confidence-medium { color: #f57c00; font-weight: 600; }
    .confidence-low    { color: #c62828; font-weight: 600; }
    .citation-card {
        background: #1e2a3a; border-left: 4px solid #4fc3f7;
        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.88rem;
        color: #e0e0e0 !important;
    }
    .citation-card strong { color: #4fc3f7 !important; }
    .citation-card em { color: #b0bec5 !important; }
    .answer-box {
        background: #0d2137; border: 1px solid #4fc3f7;
        padding: 1.2rem; border-radius: 8px;
        font-size: 1.05rem; line-height: 1.7;
        color: #e8f4fd !important;
    }
    .stat-card {
        background: #fff; border: 1px solid #e0e0e0;
        padding: 1rem; border-radius: 8px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedIntel AI")
    st.markdown("*RAG-powered medical document assistant*")
    st.divider()

    # System health
    st.markdown("### System Status")
    try:
        health = requests.get(f"{API_V1}/health", timeout=5).json()
        st.success(f"✅ API Online")
        st.metric("Documents Indexed", health.get("documents_indexed", 0))
        with st.expander("Model Info"):
            st.code(f"Embeddings: {health.get('embedding_model', 'N/A')}\nLLM: {health.get('llm_model', 'N/A')}")
    except Exception:
        st.error("❌ API Offline — start the FastAPI server")

    st.divider()

    # Upload
    st.markdown("### 📄 Upload Medical PDFs")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload clinical guidelines, research papers, or medical records.",
    )

    if uploaded_file and st.button("📤 Index Document", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                resp = requests.post(
                    f"{API_V1}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data['chunks_indexed']} chunks indexed")
                    st.session_state.uploaded_files.append(uploaded_file.name)
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    if st.session_state.uploaded_files:
        st.markdown("**Indexed files:**")
        for f in st.session_state.uploaded_files:
            st.markdown(f"📋 `{f}`")

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieval chunks (top-k)", 1, 8, 4)
    show_context = st.checkbox("Show raw citations", value=True)
    show_scores = st.checkbox("Show similarity scores", value=True)


# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔬 MedIntel AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subheader">Ask questions about your uploaded medical documents. '
    'Answers are grounded in your PDFs with citations and confidence scores.</p>',
    unsafe_allow_html=True
)

# Chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant", avatar="🏥"):
        # Answer
        st.markdown(
            f'<div class="answer-box">{entry["answer"]}</div>',
            unsafe_allow_html=True
        )
        # Confidence badge
        conf = entry["confidence"]
        label = entry["confidence_label"]
        color_class = (
            "confidence-high" if label == "High"
            else "confidence-medium" if label == "Medium"
            else "confidence-low"
        )
        bar_color = "#2e7d32" if label == "High" else "#f57c00" if label == "Medium" else "#c62828"
        st.markdown(
            f'<p>Confidence: <span class="{color_class}">{label} ({conf:.0%})</span></p>',
            unsafe_allow_html=True
        )
        st.progress(conf)

        # Citations
        if show_context and entry.get("citations"):
            with st.expander(f"📚 {len(entry['citations'])} source(s) used"):
                for i, cite in enumerate(entry["citations"], 1):
                    score_badge = f"  _(similarity: {cite['score']:.2f})_" if show_scores else ""
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<strong>[{i}] {cite["source"]} — page {cite["page"]}</strong>{score_badge}<br>'
                        f'<em>{cite["chunk_preview"][:250]}...</em>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# Query input
query = st.chat_input("Ask a medical question, e.g. 'What are the side effects of metformin?'")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_V1}/query",
                    json={"query": query, "top_k": top_k},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    confidence = data["confidence"]
                    label = data["confidence_label"]
                    citations = data["citations"]

                    # Display answer
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )

                    # Confidence
                    color_class = (
                        "confidence-high" if label == "High"
                        else "confidence-medium" if label == "Medium"
                        else "confidence-low"
                    )
                    st.markdown(
                        f'<p>Confidence: <span class="{color_class}">{label} ({confidence:.0%})</span></p>',
                        unsafe_allow_html=True
                    )
                    st.progress(confidence)

                    # Citations
                    if show_context and citations:
                        with st.expander(f"📚 {len(citations)} source(s) used"):
                            for i, cite in enumerate(citations, 1):
                                score_badge = f"  _(similarity: {cite['score']:.2f})_" if show_scores else ""
                                st.markdown(
                                    f'<div class="citation-card">'
                                    f'<strong>[{i}] {cite["source"]} — page {cite["page"]}</strong>{score_badge}<br>'
                                    f'<em>{cite["chunk_preview"][:250]}...</em>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    # Save to history
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": answer,
                        "confidence": confidence,
                        "confidence_label": label,
                        "citations": citations,
                        "timestamp": datetime.now().isoformat(),
                    })

                elif resp.status_code == 400:
                    st.warning("⚠️ " + resp.json().get("detail", "No documents indexed yet."))
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ── Quick example questions ───────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    cols = st.columns(3)
    examples = [
        "What are the contraindications mentioned?",
        "Summarise the treatment recommendations",
        "What dosage guidelines are described?",
        "What are the reported side effects?",
        "What patient populations were studied?",
        "What are the key findings of this document?",
    ]
    for i, ex in enumerate(examples):
        col = cols[i % 3]
        if col.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = ex
            st.rerun()
