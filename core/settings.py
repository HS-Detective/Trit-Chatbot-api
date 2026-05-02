# core/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 기능 A 기본값(필요시 기능별 config.yaml이 override)
DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME_FEATURE1", "FAQ")
