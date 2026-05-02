# routers/chat.py
from typing import List, Literal, Optional, Dict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.chatbot_base import RAGChatBot
from core.settings import OPENAI_API_KEY
from core.utils import load_yaml, read_text

# 맨 위 import에 추가
from typing import Any
import logging
import importlib

logger = logging.getLogger(__name__)

router = APIRouter()

# 프로젝트 루트 및 features 디렉터리
BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = BASE_DIR / "features"


def _load_bots() -> Dict[str, RAGChatBot]:
    """features/*/config.yaml을 스캔하고 동적으로 봇 클래스를 로딩하여 인스턴스 생성"""
    bots: Dict[str, RAGChatBot] = {}
    for cfg_path in FEATURES_DIR.glob("*/config.yaml"):
        cfg = load_yaml(str(cfg_path))
        fid = cfg.get("id") or cfg_path.parent.name

        chroma_dir = str((BASE_DIR / cfg["paths"]["chroma_dir"]).resolve())
        system_prompt_path = str((BASE_DIR / cfg["paths"]["system_prompt"]).resolve())

        bot_class_path = cfg.get("bot_class", "core.chatbot_base.RAGChatBot")
        try:
            module_path, class_name = bot_class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            BotClass = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(
                f"[WARN] '{fid}' 기능의 봇 클래스 '{bot_class_path}' 로딩 실패. 기본 봇을 사용합니다. 에러: {e}"
            )
            from core.chatbot_base import RAGChatBot as BotClass

        k = (cfg.get("retriever") or {}).get("k")
        bot_init_kwargs = {
            "api_key": OPENAI_API_KEY,
            "persist_directory": chroma_dir,
            "collection_name": cfg.get("collection_name") or fid,
            "llm_model": cfg["models"]["llm"],
            "embedding_model": cfg["models"]["embedding"],
            "qa_system_prompt_text": read_text(system_prompt_path),
        }

        if cfg.get("models", {}).get("embedding_dimensions"):
            bot_init_kwargs["embedding_dimensions"] = cfg["models"]["embedding_dimensions"]
        if k:
            bot_init_kwargs["retriever_k"] = int(k)

        # HSChatBot과 같은 복잡한 봇을 위해 추가 파라미터 전달
        if "paths" in cfg:
            bot_init_kwargs["paths"] = cfg["paths"]
        if "llm_params" in cfg:
            bot_init_kwargs["llm_params"] = cfg["llm_params"]

        # 기능별 커스텀 문서 프롬프트가 있으면 로드하여 전달
        if cfg.get("paths", {}).get("document_prompt"):
            doc_prompt_path = str(
                (BASE_DIR / cfg["paths"]["document_prompt"]).resolve()
            )
            bot_init_kwargs["document_prompt_template"] = read_text(doc_prompt_path)

        bots[fid] = BotClass(**bot_init_kwargs)
    return bots


BOTS = _load_bots()
DEFAULT_FEATURE = next(iter(BOTS)) if BOTS else None


# ======== 스키마 ========
class Turn(BaseModel):
    role: Literal["human", "ai"]
    content: str


class AskReq(BaseModel):
    question: str
    feature_id: Optional[str] = None  # 없으면 DEFAULT_FEATURE 사용
    chat_history: List[Turn] = []
    top_k: Optional[int] = None


class AskRes(BaseModel):
    answer: str
    sources: list
    chat_history: List[Turn]


# 스프링 호환 요청 스키마(바디 키는 동일하게 사용)
class SpringReq(BaseModel):
    question: str
    chat_history: List[Turn] = []
    top_k: Optional[int] = None
    sessionId: Optional[str] = None


# 스프링 호환 응답: reply 키로 내려줌 (RestTemplate가 reply만 읽음)
class SpringRes(BaseModel):
    reply: str
    sources: list = []
    chat_history: List[Turn] = []


# Spring mode 경로 → 실제 feature_id 매핑
# 반드시 /features에서 보이는 실제 id와 일치시켜 주세요.
PATH2ID = {
    "faq": "faq",
    "glossary": "tradeWords",  # tradeWords 기능의 id가 'tradeWords'라면 이렇게
    "hs": "hs",
    "nav": "nav",
}

# ──────────────────────────────────────────────────────────────────────────────


# 공통 호출 유틸
# ----- _call_bot 수정: 가드 + 디버그 로그 -----
def _call_bot(feature_key: str, req: SpringReq) -> SpringRes:
    if not BOTS:
        raise HTTPException(500, "No features loaded. Check features/*/config.yaml.")
    real_id = PATH2ID.get(feature_key, feature_key)
    bot = BOTS.get(real_id)
    if not bot:
        raise HTTPException(400, f"unknown feature_id: {real_id}")

    # 디버그(선택): 들어온 키 확인
    try:
        logger.debug(
            "spring payload keys=%s",
            [k for k, v in req.model_dump().items() if v is not None],
        )
    except Exception:
        pass

    # 프런트가 question으로 통일했으므로 바로 사용
    qtext = (req.question or "").strip()
    if not qtext:
        raise HTTPException(422, "Field 'question' is required")

    # (선택) 멀티턴 길이 제한: 최근 N턴만 유지하고 싶다면
    KEEP_TURNS = 10  # 최근 10턴 유지 (human+ai = 2*턴수 만큼 메시지)
    history = [t.model_dump() for t in req.chat_history][-2 * KEEP_TURNS :]

    try:
        out = bot.ask(
            question=qtext,
            chat_history=history,
            top_k=req.top_k,
            keep_last=2 * KEEP_TURNS,
        )
        return SpringRes(
            reply=out["answer"],
            sources=out["sources"],
            chat_history=out["chat_history"],
        )
    except Exception as e:
        logger.exception("Chat failed for feature=%s", real_id)
        raise HTTPException(status_code=500, detail=f"{real_id} failed: {e}")


# ======== 엔드포인트 ========
@router.get("/health", tags=["system"])
def health():
    return {"ok": True, "features": list(BOTS.keys())}


@router.get("/features", tags=["system"])
def list_features():
    return {"features": list(BOTS.keys())}


@router.post("/ask", response_model=AskRes, tags=["chat"])
def ask(req: AskReq):
    if not BOTS:
        raise HTTPException(500, "No features loaded. Check features/*/config.yaml.")

    fid = req.feature_id or DEFAULT_FEATURE
    bot = BOTS.get(fid)
    if not bot:
        raise HTTPException(400, f"unknown feature_id: {fid}")

    out = bot.ask(
        question=req.question,
        chat_history=[t.model_dump() for t in req.chat_history],
        top_k=req.top_k,
    )
    return AskRes(**out)


# ──────────────────────────────────────────────────────────────────────────────
# 스프링 호환 엔드포인트들 (/hs, /nav, /glossary, /faq)


# HS와 NAV 준비되면 위의 @router 지우고 아래 주석 해제
@router.post("/hs", response_model=SpringRes, tags=["chat"])
def chat_hs(req: SpringReq):
    return _call_bot("hs", req)


@router.post("/nav", response_model=SpringRes, tags=["chat"])
def chat_nav(req: SpringReq):
    return _call_bot("nav", req)


@router.post("/glossary", response_model=SpringRes, tags=["chat"])
def chat_glossary(req: SpringReq):
    return _call_bot("glossary", req)


@router.post("/faq", response_model=SpringRes, tags=["chat"])
def chat_faq(req: SpringReq):
    return _call_bot("faq", req)
