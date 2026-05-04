from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
import logging

from core.stats_db import engine, db, HS_CODE_MAPPING
from core.settings import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

logger = logging.getLogger(__name__)

router = APIRouter()

# Schema for Chat
class StatsChatReq(BaseModel):
    question: str

class StatsChatMeta(BaseModel):
    mode: str = "stats"
    error: bool = False

class StatsChatRes(BaseModel):
    reply: str
    meta: StatsChatMeta = StatsChatMeta()

# Schema for Chart responses
class ChartDataPoint(BaseModel):
    label: Optional[str]
    value: float
    code: Optional[str] = None


@router.post("/chat/stats", response_model=StatsChatRes, tags=["stats"])
def chat_stats(req: StatsChatReq):
    try:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
        
        # Prepare the mapping context
        mapping_context = "\n".join(HS_CODE_MAPPING[:50]) # Using first 50 lines to prevent context overflow if too large, or you can use all if small
        if len(HS_CODE_MAPPING) > 50:
            mapping_context += "\n... (and more mappings)"
            
        system_message = f"""
당신은 대한민국 무역 통계 분석 전문가입니다. 
'MTI_TRADE_STATS' 테이블을 사용하여 사용자의 질문에 답하세요.

[테이블 컬럼 설명]
- YYMM: 연월 (예: '202301')
- CTR_NAME: 국가명 (예: '중국', '미국', '베트남' 등)
- MTI_CD: MTI 품목 코드 (4단위)
- MTI_NAME: 품목명 (예: '프로세서와 컨트롤러', '평판디스플레이' 등)
- EXP_AMT: 수출 금액 (달러 단위)
- IMP_AMT: 수입 금액 (달러 단위)

[쿼리 작성 지침 - 중요]
1. 품목명 매칭: 사용자가 '반도체', '자동차' 같이 일반적인 단어로 물어보면, MTI_NAME에서 해당 단어가 포함된 모든 항목을 찾아야 합니다. 
   예: `MTI_NAME LIKE '%반도체%'` 또는 `MTI_NAME LIKE '%자동차%'`
2. 국가명 매칭: 국가명도 가급적 `LIKE`를 사용하거나 정확한 명칭을 추론하세요.
3. 데이터 부재 시: 만약 결과가 None이라면, 사용자가 물어본 키워드와 비슷한 품목명이 있는지 먼저 검색(`SELECT DISTINCT MTI_NAME ... LIKE ...`)해본 뒤 다시 쿼리하세요.
4. 답변 언어: 반드시 한국어로 답변하세요.

도메인 지식 (MTI-HS코드 매핑 참고):
{mapping_context}
"""
        
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="openai-tools",
            verbose=True,
            handle_parsing_errors=True,
            agent_executor_kwargs={"return_intermediate_steps": False}
        )
        
        res = agent_executor.invoke({
            "input": f"{system_message}\n\nUser Question: {req.question}"
        })
        
        return StatsChatRes(reply=res["output"])
    except Exception as e:
        logger.exception("Error in /chat/stats LLM generation")
        return StatsChatRes(
            reply="통계 데이터를 분석하는 도중 오류가 발생했습니다.",
            meta=StatsChatMeta(error=True)
        )

# A. Monthly Trend
@router.get("/stats/monthly-trend", response_model=List[ChartDataPoint], tags=["stats"])
def get_monthly_trend(
    metric: str = Query(..., description="예: EXP_AMT 또는 IMP_AMT"),
    country: str = Query(..., description="국가명(CTR_NAME) 또는 ALL"),
    mti_cd: str = Query(..., description="품목분류코드(MTI_CD) 또는 ALL"),
    startYm: str = Query(..., description="시작 연월 (예: 202301)"),
    endYm: str = Query(..., description="종료 연월 (예: 202312)")
):
    # Security: validate metric name to prevent SQL injection since it's used as a column name directly
    if metric not in ["EXP_AMT", "IMP_AMT"]:
        raise HTTPException(status_code=400, detail="metric must be EXP_AMT or IMP_AMT")

    query_str = f"""
    SELECT
      YYMM as label,
      SUM({metric}) as value
    FROM MTI_TRADE_STATS
    WHERE YYMM >= :startYm AND YYMM <= :endYm
    """
    
    params = {"startYm": startYm, "endYm": endYm}
    
    if country != "ALL":
        query_str += " AND CTR_NAME = :country"
        params["country"] = country
        
    if mti_cd != "ALL":
        query_str += " AND MTI_CD = :mti_cd"
        params["mti_cd"] = mti_cd
        
    query_str += " GROUP BY YYMM ORDER BY YYMM ASC"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_str), params)
            return [{"label": row.label, "value": row.value or 0} for row in result]
    except Exception as e:
        logger.exception("Error executing monthly-trend query")
        raise HTTPException(status_code=500, detail=str(e))


# B. Top Countries
@router.get("/stats/top-countries", response_model=List[ChartDataPoint], tags=["stats"])
def get_top_countries(
    metric: str = Query(..., description="예: EXP_AMT 또는 IMP_AMT"),
    year: str = Query(..., description="연도 (예: 2023)"),
    mti_cd: str = Query(..., description="품목분류코드(MTI_CD) 또는 ALL"),
    topN: int = Query(5, description="상위 N개")
):
    if metric not in ["EXP_AMT", "IMP_AMT"]:
        raise HTTPException(status_code=400, detail="metric must be EXP_AMT or IMP_AMT")

    query_str = f"""
    SELECT
      CTR_NAME as label,
      SUM({metric}) as value
    FROM MTI_TRADE_STATS
    WHERE YYMM LIKE :year_prefix
    """
    
    params = {"year_prefix": f"{year}%", "topN": topN}
    
    if mti_cd != "ALL":
        query_str += " AND MTI_CD = :mti_cd"
        params["mti_cd"] = mti_cd
        
    query_str += " GROUP BY CTR_NAME ORDER BY value DESC LIMIT :topN"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_str), params)
            return [{"label": row.label or "알수없음", "value": row.value or 0} for row in result]
    except Exception as e:
        logger.exception("Error executing top-countries query")
        raise HTTPException(status_code=500, detail=str(e))


# C. Item Share
@router.get("/stats/item-share", response_model=List[ChartDataPoint], tags=["stats"])
def get_item_share(
    metric: str = Query(..., description="예: EXP_AMT 또는 IMP_AMT"),
    year: str = Query(..., description="연도 (예: 2023)"),
    country: str = Query(..., description="국가명(CTR_NAME) 또는 ALL"),
    topN: int = Query(10, description="상위 N개")
):
    if metric not in ["EXP_AMT", "IMP_AMT"]:
        raise HTTPException(status_code=400, detail="metric must be EXP_AMT or IMP_AMT")

    query_str = f"""
    SELECT
      MTI_NAME as label,
      MTI_CD as code,
      SUM({metric}) as value
    FROM MTI_TRADE_STATS
    WHERE YYMM LIKE :year_prefix
    """
    
    params = {"year_prefix": f"{year}%", "topN": topN}
    
    if country != "ALL":
        query_str += " AND CTR_NAME = :country"
        params["country"] = country
        
    query_str += " GROUP BY MTI_NAME, MTI_CD ORDER BY value DESC LIMIT :topN"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_str), params)
            return [{"label": row.label or "알수없음", "code": row.code, "value": row.value or 0} for row in result]
    except Exception as e:
        logger.exception("Error executing item-share query")
        raise HTTPException(status_code=500, detail=str(e))


# D. Dropdown Options: Countries
@router.get("/stats/options/countries", tags=["stats"])
def get_country_options():
    """드롭다운 선택창을 위한 고유 국가 리스트 반환"""
    query_str = "SELECT DISTINCT CTR_NAME FROM MTI_TRADE_STATS WHERE CTR_NAME IS NOT NULL ORDER BY CTR_NAME ASC"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_str))
            return [row.CTR_NAME for row in result]
    except Exception as e:
        logger.exception("Error fetching country options")
        raise HTTPException(status_code=500, detail=str(e))


# E. Dropdown Options: MTI Items
@router.get("/stats/options/mti", tags=["stats"])
def get_mti_options():
    """드롭다운 선택창을 위한 고유 MTI 품목(코드+이름) 리스트 반환"""
    from core.stats_db import MTI_NAME_MAP
    
    # DB에서 현재 존재하는 MTI 코드들을 먼저 가져옵니다.
    query_str = "SELECT DISTINCT MTI_CD FROM MTI_TRADE_STATS WHERE MTI_CD IS NOT NULL ORDER BY MTI_CD ASC"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_str))
            options = []
            for row in result:
                code = row.MTI_CD
                # 마스터 맵에서 이름을 찾고, 없으면 '알 수 없음' 처리
                label = MTI_NAME_MAP.get(code, "알 수 없는 품목")
                options.append({"code": code, "label": label})
            return options
    except Exception as e:
        logger.exception("Error fetching MTI options")
        raise HTTPException(status_code=500, detail=str(e))
