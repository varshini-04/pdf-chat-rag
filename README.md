<img width="1440" height="857" alt="Screenshot 2026-07-12 at 10 23 14 PM" src="https://github.com/user-attachments/assets/39c50ab8-5d19-4cec-a7ab-dd948f29f9f9" />
---
title: Chat With Your PDF
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# 📄 Chat with your PDF — RAG Demo

**Upload any document and ask questions about it.** Answers are grounded in your file using Retrieval-Augmented Generation (RAG) — the same technique businesses use to build AI assistants over their internal docs, help centers, and product catalogs.

👉 **Live demo:** _add your Hugging Face Space link here_

![screenshot](screenshot.png) <!-- add a screenshot after deploying -->

## What it demonstrates (for clients)

- An AI assistant that answers **only from your documents** — no made-up answers
- Instant setup: upload a PDF, start asking questions
- Transparent answers: every response shows the exact source chunks it used
- Fast responses via Groq LLM inference

## How it works (for engineers)

1. **Extract** — text is pulled from the PDF (`pypdf`)
2. **Chunk** — split into ~1000-character overlapping chunks at paragraph boundaries
3. **Embed** — chunks are embedded locally with `all-MiniLM-L6-v2` (sentence-transformers)
4. **Retrieve** — each question is embedded and the top-4 chunks are found by cosine similarity
5. **Generate** — retrieved context + question are sent to a Groq-hosted LLM with a strict grounding prompt; answers stream in real time

No external vector database is needed at this scale — the index lives in memory, which keeps the demo free to run and easy to deploy. (For production workloads I use dedicated vector stores.)

**Stack:** Python · Streamlit · sentence-transformers · Groq · pypdf · NumPy

## Run locally

```bash
git clone <your-repo-url>
cd pdf-chat-rag
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here   # get a free key at console.groq.com
streamlit run app.py
```

## Deploy free on Hugging Face Spaces (10 minutes)

1. Create a free account at huggingface.co
2. Click **New Space** → name it (e.g. `chat-with-your-pdf`) → SDK: **Streamlit** → hardware: **CPU basic (free)**
3. Upload these files to the Space: `app.py`, `rag.py`, `sample_document.md`, `requirements.txt`, and this `README.md`
4. Go to **Settings → Variables and secrets** → add a secret named `GROQ_API_KEY` with your Groq API key (free at console.groq.com)
5. The Space builds automatically (first build takes a few minutes while the embedding model downloads). Done — you have a public URL to share.

> Tip: if the Groq model name is ever deprecated, change `GROQ_MODEL` at the top of `app.py` to any current model listed at console.groq.com/docs/models.

## Project structure

```
app.py               # Streamlit UI, session state, streaming LLM calls
rag.py               # RAG engine: chunking, embeddings, retrieval, prompting
sample_document.md   # preloaded sample so visitors can test in 5 seconds
requirements.txt
```

---

Built by **Gowra Sreevarshini** — AI Developer (RAG systems, LLM agents, AI-powered apps).
[GitHub](https://github.com/varshini-04) · [LinkedIn](https://www.linkedin.com/in/sreevarshini-gowra-304b95325/)
