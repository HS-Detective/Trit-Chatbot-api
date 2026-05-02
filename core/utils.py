# core/utils.py
from pathlib import Path
import yaml

def read_text(path: str, default: str = "") -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else default

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
