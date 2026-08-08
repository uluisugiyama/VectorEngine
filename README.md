# VectorEngine

A lightweight RAG (Retrieval-Augmented Generation) engine built with Python. Combines semantic search via ChromaDB and SentenceTransformers with LLM-powered question answering via Groq.

## Features

- **Document ingestion** — Upload `.txt` or `.pdf` files; text is automatically chunked and embedded.
- **Semantic search** — Query your knowledge base with natural language; results ranked by embedding distance.
- **Grounded Q&A** — Ask questions answered strictly from your uploaded documents (no hallucination).
- **Persistent storage** — ChromaDB persistence directory survives app restarts.
- **Streamlit UI** — Interactive chat interface with sidebar document management.

## Architecture

| Component | Role |
|---|---|
| `rag_engine.py` | Core engine: ingestion, search, and LLM generation |
| `app.py` | Streamlit web application |
| `test_engine.py` | Integration tests (Phases 1–5) |
| `test_app.py` | Streamlit AppTest smoke test |

### Single-Tenant Design

VectorEngine uses a **single shared knowledge base**. All uploaded documents are available to all sessions that share the same persistence directory. This is intentional for personal or single-team use.

If per-user isolation is needed in the future, the engine's `__init__` can be extended to accept a `collection_name` parameter for per-session collections.

## Setup

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file (or `.streamlit/secrets.toml`) with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

> **Never commit `.env` or `secrets.toml`** — both are in `.gitignore`.

## Running

### Local

```bash
source .venv/bin/activate
streamlit run app.py
```

### Docker

```bash
docker build -t vectorengine .
docker run -e GROQ_API_KEY=your_key_here \
           -p 8501:8501 \
           -v $(pwd)/chroma_db:/app/chroma_db \
           vectorengine
```

Open [http://localhost:8501](http://localhost:8501).

- Pass `GROQ_API_KEY` via `-e` or `--env-file .env` at runtime.
- Mount `chroma_db/` as a volume for persistent storage across container restarts.
- Omit the `-v` flag for ephemeral (non-persistent) mode.

## Testing

```bash
source .venv/bin/activate
python test_engine.py   # Phases 1–5: init, ingestion, search, generation, persistence
python test_app.py      # Streamlit UI smoke test
```
