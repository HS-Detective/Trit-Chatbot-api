# features/faq/build_db.py
from pathlib import Path
import argparse
from langchain_community.document_loaders import PyPDFLoader
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

def index_pdf(target: Path, chunk_size=1000, overlap=200):
    if not target.exists():
        raise FileNotFoundError(f"경로 없음: {target}")

    files = [target] if target.is_file() else sorted(target.glob("**/*.pdf"))
    if not files:
        raise FileNotFoundError(f"PDF가 없습니다: {target}")

    # 로드 & 분할
    docs=[]
    for f in files: docs.extend(PyPDFLoader(str(f)).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = splitter.split_documents(docs)

    # 임베딩 & Chroma
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=CFG["models"]["embedding"])
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
    p.add_argument("--pdf", help="단일 PDF 경로", default=str(script_dir / "datasets" / "manual.pdf"))
    p.add_argument("--dir", help="PDF 폴더 경로")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)
    args = p.parse_args()
    target = Path(args.pdf) if args.pdf and not args.dir else Path(args.dir) if args.dir else None
    if not target:
        raise SystemExit(" --pdf 또는 --dir 중 하나는 지정하세요.")
    index_pdf(target, args.chunk_size, args.chunk_overlap)
