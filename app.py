#!/usr/bin/env python3
"""
Streamlit Web UI for Ask-Your-PDFs (RAG).
Provides an interactive web interface to ask questions against ingested PDF documents
with strict page-level citations: [filename, p. N].

Run with:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_CHROMA_DIR,
    DEFAULT_TOP_K,
)
from src.rag import RAGService
from src.ingest import ingest_folder, ingest_single_pdf

# Set Streamlit page configuration
st.set_page_config(
    page_title="Ask-Your-PDFs | RAG with Citations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and citation badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
        font-family: monospace;
    }
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stat-card h3 {
        margin: 0;
        font-size: 1.6rem;
        color: #0F172A;
    }
    .stat-card p {
        margin: 0;
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #6366F1;
        padding: 10px 14px;
        margin-bottom: 12px;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_service():
    """Initializes and caches the RAG service instance."""
    return RAGService(chroma_dir=DEFAULT_CHROMA_DIR)


rag = get_rag_service()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    provider_options = {
        "Auto-Detect (Best Available)": "auto",
        "Google Gemini (gemini-2.5-flash)": "gemini",
        "OpenAI (gpt-4o-mini)": "openai",
        "Local Ollama": "ollama",
        "Built-in Extractive RAG (Offline)": "fallback",
    }
    selected_provider_label = st.selectbox(
        "LLM Generation Provider",
        list(provider_options.keys()),
        index=0,
        help="Select which provider generates the answers. Auto-detect chooses available API keys or falls back to built-in extractive search.",
    )
    provider_value = provider_options[selected_provider_label]

    top_k_val = st.slider(
        "Top-k Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help="Number of most similar semantic chunks passed into context.",
    )

    threshold_val = st.slider(
        "Cosine Distance Filter Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.72,
        step=0.01,
        help="Queries with top cosine distance above this threshold will answer: 'I don't know based on the provided documents.'",
    )

    st.markdown("---")
    st.markdown("### 📂 Ingest New PDF")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        if st.button("📥 Index Uploaded PDF", use_container_width=True):
            save_path = DEFAULT_DATA_DIR / uploaded_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                res = ingest_single_pdf(
                    save_path,
                    chroma_dir=DEFAULT_CHROMA_DIR,
                    model=rag.embedder,
                )
                st.success(
                    f"Indexed **{res['filename']}**: {res['pages']} pages, {res['chunks']} chunks added!"
                )
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.markdown("### 🔄 Maintenance")
    if st.button("Re-Index All PDFs from /pdfs", use_container_width=True):
        with st.spinner("Processing all documents in pdfs/..."):
            stats = ingest_folder(data_dir=DEFAULT_DATA_DIR, chroma_dir=DEFAULT_CHROMA_DIR, reset=True)
            st.success(f"Indexed {stats['files']} PDFs, {stats['chunks']} total chunks.")
            time.sleep(1)
            st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main View Header
st.markdown('<div class="main-title">📚 Ask-Your-PDFs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation (RAG) assistant with strict page-level citations in <code>[filename, p. N]</code> format.</div>',
    unsafe_allow_html=True,
)

# Statistics Overview
doc_list = rag.get_indexed_documents()
total_chunks = rag.count()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="stat-card"><h3>{len(doc_list)}</h3><p>Indexed Documents</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="stat-card"><h3>{total_chunks}</h3><p>Vector Chunks in DB</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="stat-card"><h3>all-MiniLM-L6</h3><p>Embedding Model</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Expandable Document Inventory
with st.expander(f"📑 Indexed Corpus Inventory ({len(doc_list)} files)"):
    if doc_list:
        for d in doc_list:
            pdf_path = DEFAULT_DATA_DIR / d
            size_str = f" ({round(pdf_path.stat().st_size / 1024, 1)} KB)" if pdf_path.exists() else ""
            st.markdown(f"- **{d}**{size_str}")
    else:
        st.info("No documents indexed yet. Run `python ingest.py` or use the sidebar uploader.")

# Quick Sample Questions
st.markdown("##### 💡 Try Sample Questions:")
samples = [
    "What is the formula for Scaled Dot-Product Attention?",
    "What is Low-Rank Adaptation (LoRA) and how does it freeze pre-trained weights?",
    "What are the two core components of RAG models?",
    "Why does standard attention suffer from high memory bandwidth IO bottleneck?",
    "What is the traditional recipe for French onion soup?",
]

sample_cols = st.columns(len(samples))
clicked_sample = None
for idx, s in enumerate(samples):
    label = s[:28] + "..." if len(s) > 28 else s
    if sample_cols[idx].button(label, key=f"sample_{idx}", help=s):
        clicked_sample = s

st.markdown("---")

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            st.markdown(
                "**Citations:** "
                + " ".join(
                    [
                        f'<span class="citation-badge">{c}</span>'
                        for c in msg["citations"]
                    ]
                ),
                unsafe_allow_html=True,
            )
        if "chunks" in msg and msg["chunks"]:
            with st.expander(f"🔍 View {len(msg['chunks'])} Retrieved Source Chunks"):
                for i, c in enumerate(msg["chunks"], 1):
                    meta = c.get("metadata", {})
                    dist = c.get("distance", 0.0)
                    fname = meta.get("filename", "unknown.pdf")
                    page = meta.get("page", "?")
                    c_idx = meta.get("chunk_index", 0)
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Source #{i}: <code>[{fname}, p. {page}]</code></strong> (chunk {c_idx}, cosine distance: {dist:.4f})<br>
                            <em>{c.get('text', '')}</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Handle Question Submission
user_prompt = st.chat_input("Ask a question about your ingested PDFs...")
query_to_run = clicked_sample if clicked_sample else user_prompt

if query_to_run:
    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching corpus and synthesizing answer..."):
            start_t = time.time()
            res = rag.answer_question(
                query=query_to_run,
                top_k=top_k_val,
                provider=provider_value,
                distance_threshold=threshold_val,
            )
            elapsed = time.time() - start_t

            ans = res.get("answer", "")
            citations = res.get("citations", [])
            retrieved_chunks = res.get("chunks", [])
            provider_used = res.get("provider", "")

            st.markdown(ans)

            if citations:
                st.markdown(
                    "**Citations:** "
                    + " ".join(
                        [
                            f'<span class="citation-badge">{c}</span>'
                            for c in citations
                        ]
                    ),
                    unsafe_allow_html=True,
                )

            st.caption(f"⚡ Generated via `{provider_used}` in {elapsed:.2f}s")

            if retrieved_chunks:
                with st.expander(f"🔍 View {len(retrieved_chunks)} Retrieved Source Chunks"):
                    for i, c in enumerate(retrieved_chunks, 1):
                        meta = c.get("metadata", {})
                        dist = c.get("distance", 0.0)
                        fname = meta.get("filename", "unknown.pdf")
                        page = meta.get("page", "?")
                        c_idx = meta.get("chunk_index", 0)
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <strong>Source #{i}: <code>[{fname}, p. {page}]</code></strong> (chunk {c_idx}, cosine distance: {dist:.4f})<br>
                                <em>{c.get('text', '')}</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Store in session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ans,
                    "citations": citations,
                    "chunks": retrieved_chunks,
                    "provider": provider_used,
                }
            )

