# evaluation/evaluate_hs_chatbot.py
import os
import sys
import time
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import datetime
import argparse
import csv

# --- 프로젝트 루트를 시스템 경로에 추가 ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -------------------------------------

from features.hs.chatbot import HSChatBot
from core.utils import load_yaml

def main(args):
    """HSChatBot의 성능을 평가하고 결과를 CSV로 저장합니다."""
    
    # 1. 설정 및 봇 초기화
    print("설정을 로드하고 챗봇을 초기화합니다...")
    feature_config_path = project_root / "features" / "hs" / "config.yaml"
    hs_config = load_yaml(str(feature_config_path))
    
    for key, path_str in hs_config.get("paths", {}).items():
        hs_config["paths"][key] = str((project_root / path_str).resolve())

    from core.settings import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("오류: .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    hs_config['api_key'] = OPENAI_API_KEY
    hs_config['persist_directory'] = hs_config['paths']['chroma_dir']

    bot = HSChatBot(**hs_config)
    print("챗봇 초기화 완료.")

    # 2. 평가 데이터셋 로드
    input_filename = args.input
    eval_dataset_path = project_root / "evaluation" / input_filename
    if not eval_dataset_path.exists():
        print(f"오류: 평가 데이터셋을 찾을 수 없습니다: {eval_dataset_path}")
        return
        
    df = pd.read_csv(eval_dataset_path, encoding='utf-8')
    print(f"'{eval_dataset_path.name}' 에서 {len(df)}개의 평가 데이터를 로드했습니다.")

    # 3. 평가 진행
    results = []
    total = len(df)
    elapsed_times = []

    print("\n--- 평가 시작 ---")
    for idx, row in tqdm(df.iterrows(), total=total, desc="평가 진행률"):
        start_time = time.time()

        question = row['question']
        correct_code = str(row['answer']).strip().replace(".", "").replace("-", "")

        response = bot.ask(question, evaluation_mode=True)
        answer_text = response.get("answer", "")

        elapsed = time.time() - start_time
        elapsed_times.append(elapsed)

        match = re.search(r"(\d{4}\.\d{2}-\d{4}|\d{4}\.\d{2,6}|\d{4,10})", answer_text)
        predicted_code_formatted = match.group(1) if match else ""
        predicted_code_unformatted = predicted_code_formatted.replace(".", "").replace("-", "")

        entry = {
            "정답": correct_code,
            "결과": answer_text,
            "6자리 결과": 0,
            "10자리 결과": 0,
            "질문": question,
            "소요시간(초)": round(elapsed, 2),
        }

        if predicted_code_unformatted:
            if len(correct_code) >= 6 and correct_code[:6] == predicted_code_unformatted[:6]:
                entry["6자리 결과"] = 1
            if len(correct_code) == 10 and correct_code == predicted_code_unformatted:
                entry["10자리 결과"] = 1
        
        results.append(entry)

    # 4. 상세 결과 집계
    res_df = pd.DataFrame(results)
    total_count = len(res_df)
    six_sum = res_df["6자리 결과"].sum()
    six_result_percent = (six_sum / total_count) * 100 if total_count > 0 else 0
    ten_sum = res_df["10자리 결과"].sum()
    ten_result_percent = (ten_sum / total_count) * 100 if total_count > 0 else 0
    avg_time = sum(elapsed_times) / total_count if total_count > 0 else 0

    summary = {
        "정답": "총계",
        "결과": "",
        "6자리 결과": f"{six_sum} / {total_count} ({six_result_percent:.2f}%)",
        "10자리 결과": f"{ten_sum} / {total_count} ({ten_result_percent:.2f}%)",
        "소요시간(초)": f"평균: {avg_time:.2f}초"
    }
    summary_df = pd.DataFrame([summary])
    res_df = pd.concat([res_df, summary_df], ignore_index=True)

    # 5. 마스터 로그 및 버전 관리
    result_path = project_root / 'TestResult' / 'test_record.csv'
    result_path.parent.mkdir(exist_ok=True)

    if result_path.exists():
        results_df_master = pd.read_csv(result_path)
    else:
        results_df_master = pd.DataFrame(columns=[
            "timestamp", "six_result", "ten_result", "avg_time_sec", "QA_count", "collection_name", "output_csv", 
            "llm_model", "chunk_size", "chunk_overlap", "embedding_model",
            "embedding_dimensions", "search_type", "k", "pd_temperature", "hs_temperature",
            "max_tokens", "version", "file_name",
        ])

    current_params = {
        "file_name": input_filename,
        "chunk_size": hs_config["text_splitter"]["chunk_size"],
        "chunk_overlap": hs_config["text_splitter"]["chunk_overlap"],
        "embedding_model": hs_config["models"]["embedding"],
        "embedding_dimensions": hs_config["models"]["embedding_dimensions"],
        "collection_name": hs_config["collection_name"],
        "search_type": hs_config["retriever"]["search_type"],
        "k": hs_config["retriever"]["k"],
        "llm_model": hs_config["models"]["llm"],
    }

    mask = pd.Series([True] * len(results_df_master))
    for col, value in current_params.items():
        if col in results_df_master.columns:
            mask &= (results_df_master[col].astype(str) == str(value))

    version = mask.sum() + 1
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{Path(input_filename).stem}_{timestamp_str}_v{version}.csv"
    output_path = result_path.parent / output_filename

    # 6. 마스터 로그에 새 결과 추가
    new_row = {
        "timestamp": datetime.datetime.now().date().isoformat(),
        "QA_count": total_count,
        "six_result": f"{six_result_percent:.2f}%",
        "ten_result": f"{ten_result_percent:.2f}%",
        "avg_time_sec": f"{avg_time:.2f}",
        "output_csv": output_filename,
        "version": version,
        "pd_temperature": hs_config["llm_params"]["pd_temperature"],
        "hs_temperature": hs_config["llm_params"]["hs_temperature"],
        "max_tokens": hs_config["llm_params"]["max_tokens"],
        **current_params
    }
    new_row_df = pd.DataFrame([new_row])
    if results_df_master.empty:
        results_df_master = new_row_df
    else:
        results_df_master = pd.concat([results_df_master, new_row_df], ignore_index=True)
    
    # 7. 파일 저장
    results_df_master.to_csv(result_path, index=False, encoding="utf-8-sig")
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    print("\n--- 평가 완료 ---")
    print(f"평균 응답 시간: {avg_time:.2f}초")
    print(f"6자리 정확도: {six_result_percent:.2f}% | 10자리 정확도: {ten_result_percent:.2f}%")
    print(f"상세 결과가 '{output_path}' 파일에 저장되었습니다.")
    print(f"마스터 로그가 '{result_path}' 파일에 업데이트되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HS Chatbot 성능 평가 스크립트")
    parser.add_argument("--input", type=str, default="QA.csv", help="평가에 사용할 CSV 파일 (예: QA.csv, QA_33.csv)")
    args = parser.parse_args()
    main(args)
