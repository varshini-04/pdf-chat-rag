"""
rag.py — the RAG engine.

Pipeline: extract text -> chunk it -> embed chunks -> retrieve top-k by
cosine similarity -> build a grounded prompt for the LLM.

Kept deliberately simple and dependency-light so the whole demo runs on a
free Hugging Face Space with no external vector database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Small, fast, free embedding model (~80MB). Runs locally on CPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000      # target characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks
TOP_K = 4              # how many chunks to retrieve per question


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file) -> str:
    """Extract raw text from an uploaded PDF file object."""
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph boundaries.

    Overlap preserves context that would otherwise be cut in half at a
    chunk border — a common cause of bad retrieval.
    """
    # Normalize whitespace, split into paragraphs first.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph is huge, hard-split it.
        while len(para) > chunk_size:
            head, para = para[:chunk_size], para[chunk_size - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # Start the next chunk with the tail of the previous one (overlap).
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# 3. Index: embed once, search many times
# ---------------------------------------------------------------------------

@dataclass
class DocumentIndex:
    """In-memory vector index over one document."""
    chunks: list[str]
    embeddings: np.ndarray  # shape: (num_chunks, dim), L2-normalized
    doc_name: str = "document"
    _cache: dict = field(default_factory=dict)


def build_index(text: str, model: SentenceTransformer, doc_name: str = "document") -> DocumentIndex:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No extractable text found in this document.")
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return DocumentIndex(chunks=chunks, embeddings=np.asarray(embeddings), doc_name=doc_name)


def retrieve(index: DocumentIndex, query: str, model: SentenceTransformer, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Return the top-k most similar chunks with their similarity scores."""
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    # Embeddings are normalized, so dot product == cosine similarity.
    scores = index.embeddings @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(index.chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# 4. Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question using ONLY the provided context from their document.

Rules:
- Ground every claim in the context. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that in this document." Do not guess.
- Be concise and direct. Use short paragraphs.
- When helpful, quote short phrases from the context."""


def build_messages(question: str, retrieved: list[tuple[str, float]], history: list[dict]) -> list[dict]:
    """Assemble the chat messages: system prompt + context + recent history + question."""
    context_block = "\n\n---\n\n".join(chunk for chunk, _score in retrieved)

    user_message = (
        f"Context from the document:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep the last few turns so follow-up questions work, without blowing up tokens.
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})
    return messages
