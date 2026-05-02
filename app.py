# app.py
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.chatbot_base import RAGChatBot
from core.settings import OPENAI_API_KEY
from core.utils import load_yaml, read_text

# ▶ 필요시 features/*/config.yaml 스캔하여 다수 로딩하도록 확장 가능
CFG = load_yaml("features/faq/config.yaml")

bots = {
    CFG["id"]: RAGChatBot(
        api_key=OPENAI_API_KEY,
        persist_directory=CFG["paths"]["chroma_dir"],
        collection_name=CFG["collection_name"],
        llm_model=CFG["models"]["llm"],
        embedding_model=CFG["models"]["embedding"],
        qa_system_prompt_text=read_text(CFG["paths"]["system_prompt"])
    )
}
DEFAULT_FEATURE = CFG["id"]

app = FastAPI(title="My Chatbot API")

class Turn(BaseModel):
    role: Literal["human", "ai"]
    content: str

class AskReq(BaseModel):
    question: str
    feature_id: Optional[str] = None
    chat_history: List[Turn] = []
    top_k: Optional[int] = None

class AskRes(BaseModel):
    answer: str
    sources: list
    chat_history: List[Turn]

@app.post("/ask", response_model=AskRes)
def ask(req: AskReq):
    fid = req.feature_id or DEFAULT_FEATURE
    bot = bots.get(fid)
    if not bot:
        raise HTTPException(400, f"unknown feature_id: {fid}")

    out = bot.ask(
        question=req.question,
        chat_history=[t.model_dump() for t in req.chat_history],
        top_k=req.top_k,
    )
    return AskRes(**out)

# app.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routers.chat import router as chat_router

app = FastAPI(title="My Chatbot API")
app.include_router(chat_router)  # prefix 없으면 /ask, /features, /health

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

