# 📄 Chat with your PDF — RAG Demo

**Upload any document and ask questions about it.** Answers are grounded in your file using Retrieval-Augmented Generation (RAG) — the same technique businesses use to build AI assistants over their internal docs, help centers, and product catalogs.

👉 **Live demo:** [_paste your Streamlit app link here](https://pdf-chat-rag-u8v3zzxshckjulnsegvztb.streamlit.app/)

![screenshot](<img width="1440" height="857" alt="Screenshot 2026-07-12 at 10 23 14 PM" src="https://github.com/user-attachments/assets/42c48dc0-1d93-4588-ac8d-721325af959c" />) <!-- add a screenshot after deploying -->

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

## Deploy free on Streamlit Community Cloud (10 minutes)

1. Push this project to a **public GitHub repo** (`app.py`, `rag.py`, `sample_document.md`, `requirements.txt`, `README.md`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → select your repo, branch (`main`), and main file (`app.py`).
4. Open **Advanced settings → Secrets** and add your Groq API key (free at console.groq.com):
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**. The first build takes a few minutes while the embedding model downloads. Done — you get a public `https://your-app.streamlit.app` URL to share.

> Note: the app reads the key with `os.environ.get("GROQ_API_KEY")`, which works on Streamlit Cloud. If it isn't picked up, change that line in `app.py` to `st.secrets["GROQ_API_KEY"]`.

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
