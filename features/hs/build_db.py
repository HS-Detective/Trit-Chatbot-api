# features/hs/build_db.py
import os
import shutil
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- 프로젝트 루트를 시스템 경로에 추가 ---
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -------------------------------------

from core.settings import OPENAI_API_KEY
from core.utils import load_yaml

# Load config for this feature
script_dir = Path(__file__).resolve().parent
CFG = load_yaml(str(script_dir / "config.yaml"))
CHROMA_DIR = str((project_root / CFG["paths"]["chroma_dir"]).resolve())
DATASET_PATH = project_root / CFG["paths"]["datasets_csv"]
COLLECTION_NAME = CFG["collection_name"]
EMB_MODEL = CFG["models"]["embedding"]
EMB_DIMS = CFG["models"]["embedding_dimensions"]
SPLITTER_CFG = CFG["text_splitter"]


def load_and_process_data(csv_path: Path) -> list[Document]:
    """CSV를 로드하여 LangChain Document 객체 리스트로 변환합니다."""
    print(f"데이터 로딩 시작: {csv_path}")
    if not csv_path.exists():
        print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path, dtype=str, encoding='utf-8-sig')
    except Exception as e:
        print(f"CSV 파일 로딩 중 오류 발생: {e}")
        return []

    documents = []
    for row in df.itertuples(index=False):
        hs_4 = getattr(row, 'HS_4', '')
        hs_2 = getattr(row, 'ryu', '')
        ryu_ex = getattr(row, 'ryu_ex', '')
        ho_ex = getattr(row, 'ho_ex', '')
        hs_10 = getattr(row, 'code_10', '')
        pd_name = getattr(row, 'pd_name', '')

        # 1) HS 해설서 문서
        hs_content = ryu_ex.strip() + "\n\n" + ho_ex.strip()
        if hs_content:
            documents.append(Document(
                page_content=hs_content,
                metadata={
                    "HS_4": hs_4,
                    "HS_2": hs_2,
                    "data": "hs_6",
                    "source": "hs해설서"
                }
            ))

        # 2) 10자리 품목명 문서
        pd_content = str(hs_10).strip() + "nn" + str(pd_name).strip()
        if pd_content:
            documents.append(Document(
                page_content=pd_content,
                metadata={
                    "HS_4": hs_4,
                    "HS_2": hs_2,
                    "data": "pd_name",
                    "source": "pd_name"
                }
            ))

    print(f"총 {len(documents)}개의 문서를 성공적으로 생성했습니다.")
    return documents


def build_database(chunk_size: int, chunk_overlap: int):
    """문서를 로드, 분할하고 임베딩하여 ChromaDB에 저장합니다."""
    print(f"데이터베이스 구축을 시작합니다... (Source: {DATASET_PATH})")
    documents = load_and_process_data(DATASET_PATH)
    if not documents:
        print("처리할 문서가 없어 데이터베이스 구축을 중단합니다.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SPLITTER_CFG["separators"]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"총 {len(chunks)}개의 청크로 분할했습니다.")

    # ID 생성
    hs4_counter = defaultdict(int)
    ids = []
    for doc in chunks:
        hs_4 = doc.metadata.get("HS_4", "none")
        hs4_counter[hs_4] += 1
        ids.append(f"{hs_4}_chunk_{hs4_counter[hs_4]}")

    print("임베딩 및 벡터DB 저장을 시작합니다...")
    embedding = OpenAIEmbeddings(
        model=EMB_MODEL,
        dimensions=EMB_DIMS,
        openai_api_key=OPENAI_API_KEY
    )

    if os.path.exists(CHROMA_DIR):
        print(f"기존 DB '{CHROMA_DIR}'를 삭제합니다.")
        shutil.rmtree(CHROMA_DIR)

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 배치 단위로 저장
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        print(f"{i // batch_size + 1}번째 배치 저장 중... ({len(batch_chunks)}개 청크)")
        Chroma.from_documents(
            documents=batch_chunks,
            embedding=embedding,
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            ids=batch_ids
        )

    print("\n데이터베이스 구축 완료!")
    print(f"   DB 경로: {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   총 청크 수: {len(chunks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB for HS Code feature.")
    parser.add_argument("--chunk-size", type=int, default=SPLITTER_CFG["chunk_size"])
    parser.add_argument("--chunk-overlap", type=int, default=SPLITTER_CFG["chunk_overlap"])
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    else:
        build_database(args.chunk_size, args.chunk_overlap)