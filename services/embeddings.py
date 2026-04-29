"""
Embeddings service — stores document chunks and enables semantic search.
Uses a simple TF-IDF fallback so the app works without an external vector DB.
Swap `_search_tfidf` for a real vector store (Chroma, Pinecone, etc.) as needed.
"""

import os
import json
import math
import re
from collections import defaultdict

CHUNKS_FILE = "data/chunks.json"
_chunks: list[dict] = []  # [{text, source, index}]


# ── Persistence ──────────────────────────────────────────────────────────────

def _load():
    global _chunks
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE) as f:
            _chunks = json.load(f)


def _save():
    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(_chunks, f)


_load()


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def add_document(text: str, source: str = "uploaded"):
    """Chunk a document and add to the store."""
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        _chunks.append({"text": chunk, "source": source, "index": idx})
    _save()
    return len(chunks)


def clear_documents():
    global _chunks
    _chunks = []
    _save()


# ── TF-IDF search ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str], all_docs: list[list[str]]) -> float:
    doc_len = len(doc_tokens)
    if doc_len == 0:
        return 0.0
    doc_freq = defaultdict(int, {t: doc_tokens.count(t) for t in set(doc_tokens)})
    N = len(all_docs)
    score = 0.0
    for token in set(query_tokens):
        tf = doc_freq[token] / doc_len
        df = sum(1 for d in all_docs if token in d)
        idf = math.log((N + 1) / (df + 1)) + 1
        score += tf * idf
    return score


def search_similar_chunks(query: str, top_k: int = 3) -> list[str]:
    if not _chunks:
        return []
    query_tokens = _tokenize(query)
    all_doc_tokens = [_tokenize(c["text"]) for c in _chunks]
    scores = [
        (_tfidf_score(query_tokens, dt, all_doc_tokens), i)
        for i, dt in enumerate(all_doc_tokens)
    ]
    scores.sort(reverse=True)
    return [_chunks[i]["text"] for _, i in scores[:top_k] if scores[0][0] > 0]
