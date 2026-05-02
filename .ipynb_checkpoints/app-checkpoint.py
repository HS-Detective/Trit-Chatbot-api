# app.py
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import RAGChatBot
from settings import (
    OPENAI_API_KEY, CHROMA_DIR, COLLECTION_NAME_FEATURE1,
    LLM_MODEL_FEATURE1, EMBED_MODEL_FEATURE1, PROMPT_SYSTEM_FEATURE1
)
from utils import read_text

app = FastAPI(title="My Chatbot API")

# --- 기능 1개 등록(필요 시 여러 개 추가) ---
system_prompt_1 = read_text(PROMPT_SYSTEM_FEATURE1,
    default="당신은 무쉽따 pdf 내용에 대해 알려주는 챗봇입니다.\n**제공된 문맥만을 근거하여 답변하세요.**\n외부 지식이나 사전 학습 내용은 사용하지 마세요.\n문맥을 통해 알 수 없는 질문에는 모른다고 대답하세요.\n\n{context}\n")

bots = {
    "feature1": RAGChatBot(
        api_key=OPENAI_API_KEY,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME_FEATURE1,
        llm_model=LLM_MODEL_FEATURE1,
        embedding_model=EMBED_MODEL_FEATURE1,
        qa_system_prompt=system_prompt_1
    )
}
DEFAULT_FEATURE = "feature1"

# --- 스키마 ---
class Turn(BaseModel):
    role: Literal["human", "ai"]
    content: str

class AskReq(BaseModel):
    question: str
    feature_id: Optional[str] = None  # "feature1" 등
    chat_history: List[Turn] = []
    top_k: Optional[int] = None

class AskRes(BaseModel):
    answer: str
    sources: list
    chat_history: List[Turn]

# --- 엔드포인트 ---
@app.post("/ask", response_model=AskRes)
def ask(req: AskReq):
    fid = req.feature_id or DEFAULT_FEATURE
    bot = bots.get(fid)
    if not bot:
        raise HTTPException(400, f"unknown feature_id: {fid}")

    out = bot.ask(
        question=req.question,
        chat_history=[t.model_dump() for t in req.chat_history],
        top_k=req.top_k
    )
    return AskRes(**out)
