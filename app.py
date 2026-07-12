"""
app.py — "Chat with your PDF" demo (Streamlit + RAG + Groq).

Run locally:
    export GROQ_API_KEY=your_key
    streamlit run app.py

On Hugging Face Spaces, set GROQ_API_KEY as a Space secret.
"""

import os
from pathlib import Path

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

import rag

# You can swap this for any current Groq chat model.
GROQ_MODEL = "llama-3.3-70b-versatile"
SAMPLE_DOC_PATH = Path(__file__).parent / "sample_document.md"

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="centered")

# --- Light styling: keep it clean, let the demo be the hero -----------------
st.markdown(
    """
    <style>
      .block-container { max-width: 780px; }
      .stChatMessage { border-radius: 12px; }
      div[data-testid="stFileUploader"] { border-radius: 12px; }
      .subtitle { color: #6b7280; font-size: 1.02rem; margin-top: -0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached resources --------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(rag.EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it as an environment variable / Space secret.")
        st.stop()
    return Groq(api_key=api_key)


# --- Session state -----------------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = None
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"/"assistant", "content": str}]
if "loaded_doc" not in st.session_state:
    st.session_state.loaded_doc = None


def load_document(text: str, doc_name: str):
    embedder = load_embedder()
    with st.spinner(f"Indexing “{doc_name}” — chunking and embedding..."):
        st.session_state.index = rag.build_index(text, embedder, doc_name=doc_name)
    st.session_state.history = []
    st.session_state.loaded_doc = doc_name


# --- Sidebar: document controls ----------------------------------------------

with st.sidebar:
    st.header("Your document")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.session_state.loaded_doc != uploaded.name:
        try:
            text = rag.extract_text_from_pdf(uploaded)
            if not text:
                st.warning("No selectable text found — this PDF may be a scanned image.")
            else:
                load_document(text, uploaded.name)
                st.success(f"Ready: {uploaded.name}")
        except Exception as e:
            st.error(f"Couldn't read that PDF: {e}")

    if st.button("Use the sample document instead"):
        sample_text = SAMPLE_DOC_PATH.read_text(encoding="utf-8")
        load_document(sample_text, "Aurora Coffee Machine — User Manual (sample)")
        st.success("Sample document loaded.")

    if st.session_state.index:
        st.caption(
            f"Loaded: **{st.session_state.index.doc_name}** · "
            f"{len(st.session_state.index.chunks)} chunks indexed"
        )
        if st.button("Clear document"):
            st.session_state.index = None
            st.session_state.history = []
            st.session_state.loaded_doc = None
            st.rerun()

    st.divider()
    st.caption(
        "How it works: your document is split into chunks and embedded. "
        "Each question retrieves the most relevant chunks, and the AI answers "
        "using only that retrieved context (RAG)."
    )


# --- Main: header + chat -------------------------------------------------------

st.title("📄 Chat with your PDF")
st.markdown(
    '<p class="subtitle">Upload any document and ask questions about it — '
    "answers are grounded in your file using Retrieval-Augmented Generation.</p>",
    unsafe_allow_html=True,
)

if st.session_state.index is None:
    st.info("Upload a PDF in the sidebar, or click **Use the sample document** to try it in 5 seconds.")

# Render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about the document...")

if question:
    if st.session_state.index is None:
        st.warning("Load a document first (sidebar).")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    embedder = load_embedder()
    retrieved = rag.retrieve(st.session_state.index, question, embedder)
    messages = rag.build_messages(question, retrieved, st.session_state.history)

    client = load_groq_client()
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,   # low temperature = factual, grounded answers
                max_tokens=1024,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content or ""
                answer += delta
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
        except Exception as e:
            placeholder.error(f"LLM request failed: {e}")
            st.stop()

        # Show the evidence — great for demos and client trust.
        with st.expander("Sources — retrieved chunks used for this answer"):
            for i, (chunk, score) in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}** · similarity {score:.2f}")
                st.text(chunk[:600] + ("..." if len(chunk) > 600 else ""))

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})
