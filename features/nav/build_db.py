import os
import re
import shutil
import pandas as pd
import argparse
from pathlib import Path

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
CHROMA_DIR = CFG["paths"]["chroma_dir"]
DATASET_PATH = script_dir / "datasets" / Path(CFG["paths"]["datasets_csv"]).name
COLLECTION_NAME = CFG["collection_name"]
EMBEDDING_MODEL = CFG["models"]["embedding"]


def load_and_process_data(csv_path: str) -> list[Document]:
    print(f"데이터 로딩 시작: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로: {csv_path}")
        return []
    except Exception as e:
        print(f"CSV 파일 로딩 중 오류 발생: {e}")
        return []

    documents = []
    for index, row in df.iterrows():
        try:
            menu_path = row.get("menu_path", "")
            desc_block_full = str(row.get("menu_description_block", ""))
            qa_block = str(row.get("menu_qa", ""))

            desc_block = re.split(
                r"(지도 구성은|표 구성은|그래프 구성은)", desc_block_full
            )[0].strip()

            url_match = re.search(r"🔗\s*(https?://[^\s\n]+)", qa_block)
            url = url_match.group(1) if url_match else None

            login_required = (
                "유료회원"
                if "기업회원(유료회원사)" in qa_block
                else (
                    "일반회원"
                    if "일반 회원 로그인이 필요합니다" in qa_block
                    else "비회원가능"
                )
            )

            table_match = re.search(
                r"(표 구성은.+?)(?:지도 구성은|그래프 구성은|$)",
                desc_block_full,
                re.DOTALL,
            )
            table_info = table_match.group(1).strip() if table_match else ""

            graph_match = re.search(
                r"(그래프 구성은.+?)(?:지도 구성은|표 구성은|$)",
                desc_block_full,
                re.DOTALL,
            )
            graph_info = graph_match.group(1).strip() if graph_match else ""

            map_match = re.search(
                r"(지도 구성은.+?)(?:그래프 구성은|표 구성은|$)",
                desc_block_full,
                re.DOTALL,
            )
            map_info = map_match.group(1).strip() if map_match else ""

            description_enriched = desc_block
            qa_pairs = re.findall(
                r"질문:\s*(.*?)\n답변:\s*(.*?)(?=\n질문:|$)", qa_block, re.DOTALL
            )
            if qa_pairs:
                description_enriched += "\n\n[자주 묻는 질문]\n"
                for q, a in qa_pairs:
                    description_enriched += f"Q. {q.strip()}\nA. {a.strip()}\n"

            doc = Document(
                page_content=description_enriched,
                metadata={
                    "menu_path": menu_path,
                    "table_info": table_info,
                    "graph_info": graph_info,
                    "map_info": map_info,
                    "url": url,
                    "login_required": login_required,
                    "source": "경로안내",
                },
            )
            documents.append(doc)
        except Exception as e:
            print(f"{index}번 행 처리 중 오류 발생: {e}\n데이터: {row.to_dict()}")

    print(f"총 {len(documents)}개의 문서를 성공적으로 생성했습니다.")
    return documents


def build_database(chunk_size=1000, chunk_overlap=100):
    print(f"데이터베이스 구축을 시작합니다... (Source: {DATASET_PATH})")
    documents = load_and_process_data(DATASET_PATH)
    if not documents:
        print("처리할 문서가 없어 데이터베이스 구축을 중단합니다.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    print(f"총 {len(docs)}개의 청크로 분할했습니다.")

    print("임베딩 및 벡터DB 저장을 시작합니다...")
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    if os.path.exists(CHROMA_DIR):
        print(f"기존 DB '{CHROMA_DIR}'를 삭제합니다.")
        shutil.rmtree(CHROMA_DIR)

    os.makedirs(CHROMA_DIR, exist_ok=True)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print("데이터베이스 구축 완료!")
    print(f"   DB 경로: {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ChromaDB for a specific feature."
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    else:
        build_database(args.chunk_size, args.chunk_overlap)
