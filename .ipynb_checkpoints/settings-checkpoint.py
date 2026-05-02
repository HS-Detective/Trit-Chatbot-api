# settings.py
import os
from dotenv import load_dotenv

load_dotenv()  # .env 읽기

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chromadb")
COLLECTION_NAME_FEATURE1 = os.getenv("COLLECTION_NAME_FEATURE1", "test_0724_1")

# (필요 시) 기능별 모델/임베딩/프롬프트 파일 경로도 .env로 분리 가능
LLM_MODEL_FEATURE1 = os.getenv("LLM_MODEL_FEATURE1", "gpt-3.5-turbo")
EMBED_MODEL_FEATURE1 = os.getenv("EMBED_MODEL_FEATURE1", "text-embedding-3-large")
PROMPT_SYSTEM_FEATURE1 = os.getenv("PROMPT_SYSTEM_FEATURE1", "./prompts/system.txt")
