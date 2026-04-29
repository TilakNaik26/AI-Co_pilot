from fastapi import APIRouter
from pydantic import BaseModel
from services.llm_service import ask_llm

router = APIRouter()


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


@router.post("/chat")
def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]
    response = ask_llm(request.message, conversation_history=history)
    return {"response": response}
