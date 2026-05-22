"""
frontend/streamlit_app.py
Modern MedIntel AI frontend
"""

import requests
import streamlit as st

# ── CONFIG ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="MedIntel AI",
    page_icon="🧠",
    layout="wide",
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "auto_query" not in st.session_state:
    st.session_state.auto_query = None

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* Main */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Header */
.main-header {
    font-size: 3rem;
    font-weight: 800;
    color: #3b82f6;
    margin-bottom: 0.2rem;
}

.subheader {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Chat Cards */
.chat-user {
    background: #1e293b;
    padding: 1rem;
    border-radius: 14px;
    margin: 1rem 0;
    border: 1px solid #334155;
}

.chat-assistant {
    background: #0f172a;
    padding: 1rem;
    border-radius: 14px;
    margin: 1rem 0;
    border: 1px solid #2563eb;
}

/* Answer Box */
.answer-box {
    background: linear-gradient(180deg, #0f172a, #111827);
    border: 1px solid #2563eb;
    padding: 1.2rem;
    border-radius: 14px;
    line-height: 1.8;
    font-size: 1rem;
    color: #f8fafc !important;
}

/* Citation Cards */
.citation-card {
    background: #111827;
    border: 1px solid #1e40af;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.8rem;
}

/* Confidence */
.confidence-good {
    color: #22c55e;
    font-weight: 700;
}

.confidence-medium {
    color: #eab308;
    font-weight: 700;
}

.confidence-low {
    color: #ef4444;
    font-weight: 700;
}

/* Buttons */
.stButton button {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    transition: 0.2s ease;
}

.stButton button:hover {
    border-color: #3b82f6 !important;
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("## 🧠 MedIntel AI")
    st.caption("RAG-powered medical document assistant")

    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health").json()

        if health.get("status") == "ok":
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")

    except Exception:
        st.error("❌ Backend not reachable")

    st.markdown("### Documents Indexed")
    st.markdown(
        f"# {len(st.session_state.indexed_files)}"
    )

    st.divider()

    # Upload section
    st.markdown("## 📄 Upload Medical PDFs")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.info(f"Selected: {uploaded_file.name}")

        if st.button("📥 Index Document", use_container_width=True):

            with st.spinner("Processing PDF and generating embeddings..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "application/pdf"
                    )
                }

                try:
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files=files,
                        timeout=300
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.success(
                            f"✅ {data['chunks_indexed']} chunks indexed"
                        )

                        if uploaded_file.name not in st.session_state.indexed_files:
                            st.session_state.indexed_files.append(
                                uploaded_file.name
                            )

                    else:
                        st.error(
                            f"Upload failed: {response.text}"
                        )

                except Exception as e:
                    st.error(f"Upload error: {e}")

    if st.session_state.indexed_files:

        st.markdown("### Indexed files:")

        for f in st.session_state.indexed_files:
            st.code(f)

    st.divider()

    # Retrieval settings
    st.markdown("## ⚙️ Settings")

    top_k = st.slider(
        "Retrieval chunks (top-k)",
        min_value=1,
        max_value=8,
        value=4
    )

# ── MAIN PAGE ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">🧠 MedIntel AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subheader">'
    'AI-powered medical document intelligence with semantic retrieval and grounded answers.'
    '</div>',
    unsafe_allow_html=True
)

# ── EXAMPLE QUESTIONS ─────────────────────────────────────────────────────────

st.markdown("### 💡 Example Questions")

example_questions = [
    "What is concentration-dependent killing?",
    "Explain bacteriostatic vs bactericidal drugs.",
    "What side effects are mentioned?",
    "Summarize antimicrobial classifications.",
]

cols = st.columns(2)

for idx, q in enumerate(example_questions):

    with cols[idx % 2]:

        if st.button(q, use_container_width=True):
            st.session_state.auto_query = q

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────────────

query = st.chat_input(
    "Ask a medical question..."
)

if st.session_state.auto_query:
    query = st.session_state.auto_query
    st.session_state.auto_query = None

# ── QUERY HANDLING ────────────────────────────────────────────────────────────

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Retrieving relevant medical context..."):

            try:

                response = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "query": query,
                        "top_k": top_k,
                    },
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer generated."
                    )

                    confidence = float(
                        data.get("confidence", 0)
                    )

                    citations = data.get(
                        "citations",
                        []
                    )

                    # Answer box
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )

                    # Confidence
                    confidence_pct = int(confidence * 100)

                    if confidence_pct >= 75:
                        badge = "🟢 High"
                        cls = "confidence-good"

                    elif confidence_pct >= 50:
                        badge = "🟡 Medium"
                        cls = "confidence-medium"

                    else:
                        badge = "🔴 Low"
                        cls = "confidence-low"

                    st.markdown(
                        f'<p class="{cls}">Confidence: {badge} ({confidence_pct}%)</p>',
                        unsafe_allow_html=True
                    )

                    # Citations
                    if citations:

                        with st.expander(
                            f"📚 {len(citations)} source(s) used",
                            expanded=False
                        ):

                            for idx, c in enumerate(citations, start=1):

                                st.markdown(
                                    f"""
<div class="citation-card">
<b>[{idx}] {c.get('source', 'Unknown')}</b> — page {c.get('page', '?')}<br><br>
Similarity: {round(c.get('score', 0), 3)}
</div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:

                    st.error(
                        f"Error {response.status_code}: {response.text}"
                    )

            except Exception as e:

                st.error(f"Connection error: {e}")