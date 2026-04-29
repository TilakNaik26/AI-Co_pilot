import os
from anthropic import Anthropic
from services.embeddings import search_similar_chunks

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are an AI Life Copilot — a smart, empathetic personal assistant.
You help users with life planning, productivity, personal growth, and answering questions.
When the user has uploaded documents, relevant context from those documents will be provided.
Be concise, insightful, and actionable in your responses."""


def ask_llm(message: str, conversation_history: list = None) -> str:
    """
    Send a message to Claude, optionally with RAG context from uploaded docs.
    conversation_history: list of {"role": "user"|"assistant", "content": str}
    """
    # Try to find relevant context from uploaded documents
    rag_context = ""
    try:
        chunks = search_similar_chunks(message, top_k=3)
        if chunks:
            rag_context = "\n\n[Relevant context from your documents]:\n" + "\n---\n".join(chunks)
    except Exception:
        pass  # No embeddings yet — that's fine

    # Build messages list
    messages = []
    if conversation_history:
        messages.extend(conversation_history)

    user_content = message
    if rag_context:
        user_content = f"{message}{rag_context}"

    messages.append({"role": "user", "content": user_content})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text
