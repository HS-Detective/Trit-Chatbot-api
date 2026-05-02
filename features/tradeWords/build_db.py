# features/tradeWords/build_db.py
from pathlib import Path
import argparse, json
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- 프로젝트 루트를 시스템 경로에 추가 ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -------------------------------------

from core.settings import OPENAI_API_KEY
from core.utils import load_yaml

script_dir = Path(__file__).resolve().parent
CFG = load_yaml(str(script_dir / 'config.yaml'))
CHROMA_DIR = CFG["paths"]["chroma_dir"]
COLLECTION_NAME = CFG["collection_name"]
EMB_MODEL = CFG["models"]["embedding"]

def read_json_docs(path: Path) -> List[Document]:
    files: List[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("**/*.json"))
    else:
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")

    docs: List[Document] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))

        # case1) 리스트 형태: [{"content": "...", "metadata": {...}}, ...]
        if isinstance(data, list):
            for item in data:
                content = item.get("content") or item.get("text") or ""
                meta = item.get("metadata") or {}
                if not content:
                    continue
                meta.setdefault("source", str(f))
                docs.append(Document(page_content=content, metadata=meta))
        # case2) dict 형태일 때 간단 매핑 (필요시 수정)
        elif isinstance(data, dict):
            content = data.get("content") or data.get("text") or json.dumps(data, ensure_ascii=False)
            docs.append(Document(page_content=content, metadata={"source": str(f)}))
    return docs

def main(target: Path, chunk_size=2000, overlap=100):
    docs = read_json_docs(target)
    if not docs:
        raise SystemExit(f"문서가 없습니다: {target}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMB_MODEL)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    ids = [f"{COLLECTION_NAME}_{i}" for i in range(len(splits))]
    db.add_documents(splits, ids=ids)

    print(f"[OK] Indexed {len(splits)} chunks → {CHROMA_DIR} (collection={COLLECTION_NAME})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", help="단일 JSON 파일 경로", default=str(script_dir / "datasets" / "tradeWords.json"))
    p.add_argument("--dir", help="JSON 폴더 경로")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--chunk-overlap", type=int, default=100)
    args = p.parse_args()

    target = Path(args.json) if args.json and not args.dir else Path(args.dir) if args.dir else None
    if not target:
        raise SystemExit("--json 또는 --dir 중 하나는 지정하세요.")
    main(target, args.chunk_size, args.chunk_overlap)
