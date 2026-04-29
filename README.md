# AI Life Copilot

A FastAPI + Claude-powered personal AI assistant with document RAG and a clean dark-themed UI.

## Project Structure

```
ai-copilot/
├── main.py                  # FastAPI app entry point
├── requirements.txt
├── static/
│   └── index.html           # Frontend UI (served by FastAPI)
├── api/
│   ├── __init__.py
│   ├── chat_routes.py       # POST /api/chat
│   └── upload_routes.py     # POST /api/upload, DELETE /api/documents
├── services/
│   ├── __init__.py
│   ├── llm_service.py       # Claude API wrapper with RAG
│   └── embeddings.py        # TF-IDF document store
└── data/                    # Auto-created: uploaded files + chunk index
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
uvicorn main:app --reload
```

Open http://localhost:8000

## Features

- **Chat** — multi-turn conversation with Claude
- **Document upload** — drag-and-drop TXT / PDF / Markdown files
- **RAG** — uploaded docs are chunked, indexed, and retrieved automatically when relevant
- **Clean UI** — dark-themed, responsive chat interface

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Serves frontend |
| POST | /api/chat | Send a message `{message, history[]}` |
| POST | /api/upload | Upload a file (multipart) |
| DELETE | /api/documents | Clear all indexed documents |
| GET | /health | Health check |

## Upgrading the vector store

The default embeddings service uses TF-IDF (no external dependencies).
To swap in a real vector DB, replace `search_similar_chunks` in
`services/embeddings.py` with your preferred client (Chroma, Pinecone, etc.).
