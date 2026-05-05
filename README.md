# Trit - AI 기반 무역 인사이트 및 챗봇 웹 서비스 

![Trit Logo](./Trit_logo.png)

## 프로젝트 개요
**Trit (AI 기반 무역 인사이트 및 챗봇 웹 서비스)**의 핵심 두뇌 역할을 하는 **AI 코어 파이썬 서버**입니다.
이 프로젝트는 메인 웹 서비스(Java/Spring Boot)를 보조하여, 무역 관련 질의응답(FAQ, 용어 사전, HS 코드, 내비게이션 등)을 처리하기 위한 맞춤형 RAG(Retrieval-Augmented Generation) 챗봇 모델 서빙 및 AI 데이터 처리 기능을 전담합니다. 

사용자의 복잡한 질문에 대해 LLM 프롬프트 처리와 정교한 답변 생성을 담당하며, 데이터 예측(Predict) 모델 서빙 및 향후 n8n 워크플로우와의 연동을 통해 더욱 확장된 AI 파이프라인을 구축하는 것을 목표로 합니다. 메인 서버와는 역할을 분리하여 유연하고 확장 가능한 마이크로서비스 아키텍처의 한 축을 담당합니다.

## 주요 기능
- **도메인별 다중 RAG 챗봇 모델 서빙 (`features` 기반 동적 로딩)**
  - `FAQ`: 무역 통관, 관세 등 자주 묻는 질문에 대한 AI 응답
  - `Glossary (tradeWords)`: 방대한 무역 전문 용어 사전 기반 검색 및 해설
  - `HS Code`: 사용자 질의 기반 HS 코드 예측 및 품목 분류 안내
  - `Navigation`: Trit 서비스 내 최적의 메뉴 및 가이드 추천
- **메인 웹 서버(Spring Boot) 맞춤형 REST API 제공**
  - 메인 서버와의 원활한 연동을 위해 스프링 서버의 요청 스키마(`SpringReq`)에 최적화된 전용 엔드포인트 제공(`/hs`, `/nav`, `/glossary`, `/faq`)
  - 자연스러운 대화 문맥 유지를 위한 챗 히스토리 관리 및 정보 신뢰도를 위한 문서 출처(Sources) 반환 기능
- **AI 데이터 텍스트 처리 및 파이프라인 연동**
  - 벡터 데이터베이스(ChromaDB)를 활용한 빠르고 정확한 유사도 검색
  - 도메인별 시스템 프롬프트 및 문서 프롬프트 템플릿의 동적 관리
  - n8n 워크플로우를 통한 외부 API 및 데이터 전처리 파이프라인 통합 지원

## 기술 스택
### Backend & Framework
- **Language**: Python 3
- **Web Framework**: FastAPI (비동기 처리 및 빠른 API 개발)
- **Server**: Uvicorn

### AI & Machine Learning
- **LLM & RAG Framework**: LangChain (<0.3), LangChain-OpenAI
- **Vector Database**: ChromaDB (임베딩 벡터 저장 및 검색)
- **Data Processing**: Pandas, NumPy
- **Document Parsing & Scraping**: PyPDF, BeautifulSoup4 (bs4), Pillow

### Utilities
- **Data Validation & Settings**: Pydantic, python-dotenv, PyYAML
- **Database Connection**: PyMySQL

## 프로젝트 폴더 구조
```text
C:\diane\chat\
├── core/                  # 핵심 AI 로직 및 공통 유틸리티
│   ├── chatbot_base.py    # 통합 RAG 챗봇 엔진 클래스 (RAGChatBot)
│   ├── settings.py        # 환경 변수 (OPENAI_API_KEY 등) 관리
│   └── utils.py           # YAML 로딩 및 파일 읽기 공통 함수
├── features/              # 도메인별 챗봇 모듈 (디렉터리 스캔을 통해 동적 로딩)
│   ├── faq/               # FAQ 챗봇 (크로마DB, 프롬프트, 데이터셋, 설정)
│   ├── hs/                # HS 코드 전용 챗봇
│   ├── nav/               # 서비스 내비게이션 챗봇
│   └── tradeWords/        # 무역 전문 용어(Glossary) 챗봇
├── routers/               # FastAPI 엔드포인트 라우터 정의
│   └── chat.py            # 메인 서버와의 통신을 위한 엔드포인트 (/faq, /hs, /nav 등)
├── scripts/               # 초기 데이터베이스 구축 등 유틸리티 스크립트
│   └── build_db.py        
├── evaluation/            # AI 응답 품질 및 시스템 평가 스크립트 모음
├── app.py                 # FastAPI 애플리케이션 진입점 및 전역 라우터 설정
├── main.py                # 콘솔 텍스트 기반의 챗봇 테스트 실행 스크립트
└── requirements.txt       # 프로젝트 패키지 의존성 목록
```

## 설정 및 실행 방법

### 1. 가상환경 설정 및 활성화
안전한 패키지 관리를 위해 가상환경 생성을 권장합니다.
```bash
# 가상환경 생성 (Windows/Mac 공통)
python -m venv .venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
source .venv/bin/activate
```

### 2. 패키지 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
프로젝트 최상단(루트) 경로에 `.env` 파일을 생성하고, 발급받은 API 키를 입력합니다.
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 서버 실행
FastAPI 서버를 **8001번 포트**에서 실행합니다. (포트 8001은 메인 Spring Boot 서버와의 충돌을 방지하기 위함입니다.)
```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```
- 서버가 정상적으로 구동되면 브라우저에서 `http://127.0.0.1:8001/docs`에 접속하여 Swagger UI를 통해 실시간 API 테스트를 진행할 수 있습니다.

*(선택) API를 거치지 않고 터미널에서 간단하게 챗봇 로직을 테스트하려면 아래 명령어를 사용하세요:*
```bash
python main.py
```

## 메인 서버(Spring Boot)와의 연동 아키텍처 안내

본 파이썬 AI 코어 서버는 **8001번 포트**에서 동작하며, 메인 웹 서버(Java/Spring Boot)와 **RESTful API**를 통해 비동기적으로 통신합니다.

- **데이터 처리 흐름**:
  1. 사용자가 메인 웹 서버(Trit UI)에 챗봇 질문 전송
  2. 메인 서버는 사용자의 위치 및 맥락을 파악하여, 이 파이썬 서버의 적절한 도메인 엔드포인트(예: `/faq`, `/hs`)로 `POST` API 요청
  3. 파이썬 서버는 요청받은 텍스트를 임베딩하여 ChromaDB에서 관련 지식을 검색하고, LangChain 체인을 통해 LLM 기반의 최적화된 답변 생성
  4. 파이썬 서버가 답변 텍스트(`reply`)와 참고 문서 출처(`sources`)를 JSON 포맷으로 응답
  5. 메인 서버가 이를 수신하여 클라이언트 화면에 최종 출력
  
- **주요 연동 엔드포인트**:
  - `POST /hs` : HS 코드 검색 및 분류 가이드
  - `POST /nav` : 서비스 내비게이션 안내
  - `POST /glossary` : 무역 전문 용어 해설
  - `POST /faq` : 기타 무역 관련 자주 묻는 질문
  
- **통신 데이터 스키마 예시**:
  - **요청 (Request - `SpringReq`)**
    ```json
    {
      "question": "수입 통관 절차에 대해 알려주세요.",
      "chat_history": [{"role": "human", "content": "이전 질문 내역"}],
      "top_k": 3,
      "sessionId": "session-1234"
    }
    ```
  - **응답 (Response - `SpringRes`)**
    ```json
    {
      "reply": "수입 통관 절차는 크게 물품 반입, 수입 신고, 세관 심사, 관세 납부, 물품 반출의 단계로 진행됩니다...",
      "sources": [
        {
          "page_content": "관세청 수입 통관 규정...",
          "metadata": {"source": "customs_guide.pdf"}
        }
      ],
      "chat_history": [...]
    }
    ```
