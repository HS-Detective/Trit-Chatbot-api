import os
import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import asyncio

# --- 프로젝트 루트를 시스템 경로에 추가 ---
# 스크립트의 현재 위치를 기준으로 프로젝트 루트를 결정합니다.
try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    # __file__이 정의되지 않은 경우 (예: 대화형 환경)
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# -------------------------------------

# --- 필요한 모듈 임포트 ---
# 환경 변수 및 설정 로더
from dotenv import load_dotenv
from core.utils import load_yaml
# 챗봇 클래스
from features.nav.chatbot import NavChatBot

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 주요 경로 및 설정 ---
EVAL_DIR = project_root / "evaluation"
FEATURE_DIR = project_root / "features" / "nav"
OUTPUT_CSV_PATH = EVAL_DIR / "nav_evaluation_results.csv"
EVAL_DATASET_PATH = EVAL_DIR / "menu_qa_set.csv"

# Nav 챗봇 설정 로드
CFG = load_yaml(str(FEATURE_DIR / "config.yaml"))

def initialize_chatbot():
    """NavChatBot을 초기화하고 반환합니다."""
    print("챗봇 초기화를 시작합니다...")
    try:
        # 설정 파일에서 필요한 값들을 가져옵니다.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

        # NavChatBot 초기화에 필요한 매개변수들을 준비합니다.
        bot_params = {
            "api_key": api_key,
            "persist_directory": str(project_root / CFG["paths"]["chroma_dir"]),
            "collection_name": CFG["collection_name"],
            "llm_model": CFG["models"]["llm"],
            "embedding_model": CFG["models"]["embedding"],
            "retriever_k": CFG["retriever"]["k"],
            "qa_system_prompt_text": (project_root / CFG["paths"]["system_prompt"]).read_text(encoding="utf-8"),
            "document_prompt_template": (project_root / CFG["paths"]["document_prompt"]).read_text(encoding="utf-8"),
        }
        
        chatbot = NavChatBot(**bot_params)
        print("챗봇 초기화 완료.")
        return chatbot
    except Exception as e:
        print(f"챗봇 초기화 중 오류 발생: {e}")
        return None

def run_evaluation(chatbot, dataset_path):
    """평가 데이터셋을 사용하여 챗봇의 성능을 평가합니다."""
    if not chatbot:
        print("챗봇이 초기화되지 않아 평가를 진행할 수 없습니다.")
        return None

    print(f"평가 데이터셋 로딩: {dataset_path}")
    try:
        eval_df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"오류: 평가 데이터 파일을 찾을 수 없습니다. 경로: {dataset_path}")
        return None
    
    results = []
    total_time = 0
    total_score = 0

    print("챗봇 성능 평가를 시작합니다...")
    # tqdm을 사용하여 진행 상황을 표시합니다.
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="평가 진행"):
        question = row["question"]
        expected_answer = str(row["answer"])

        start_time = time.time()
        
        # 챗봇에게 질문하고 답변을 받습니다.
        # NavChatBot의 ask 메서드는 동기적으로 호출할 수 있습니다.
        response = chatbot.ask(question=question)
        
        end_time = time.time()

        result_text = response.get("answer", "").strip()
        time_taken = end_time - start_time
        
        # 점수 계산: 정답(answer)이 결과(result)에 포함되어 있으면 1점
        score = 1 if expected_answer in result_text else 0
        
        total_time += time_taken
        total_score += score

        results.append({
            "question": question,
            "result": result_text,
            "answer": expected_answer,
            "score": score,
            "time_taken": f"{time_taken:.2f}s"
        })

    print("성능 평가 완료.")
    
    # 결과 데이터프레임 생성
    results_df = pd.DataFrame(results)
    
    # 요약 정보 계산
    num_questions = len(eval_df)
    if num_questions > 0:
        avg_score = total_score / num_questions
        avg_time = total_time / num_questions
    else:
        avg_score = 0
        avg_time = 0

    # 요약 정보를 담은 데이터프레임 생성
    summary_df = pd.DataFrame({
        "question": ["--- 요약 ---"],
        "result": [""],
        "answer": [""],
        "score": [f"총점: {total_score}/{num_questions}"],
        "time_taken": [f"평균: {avg_time:.2f}s"]
    })
    
    summary_df_2 = pd.DataFrame({
        "question": [f"평균 점수: {avg_score:.2f}"],
        "result": [""],
        "answer": [""],
        "score": [""],
        "time_taken": [""]
    })

    # 결과와 요약 정보를 합칩니다.
    final_df = pd.concat([results_df, summary_df, summary_df_2], ignore_index=True)
    
    return final_df

def main():
    """메인 실행 함수"""
    chatbot = initialize_chatbot()
    
    if chatbot:
        final_results = run_evaluation(chatbot, EVAL_DATASET_PATH)
        
        if final_results is not None:
            # 결과를 CSV 파일로 저장
            final_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"\n평가 결과가 다음 파일에 저장되었습니다: {OUTPUT_CSV_PATH}")
            print("\n--- 최종 요약 ---")
            summary = final_results.iloc[-2:].to_string(header=False, index=False)
            print(summary)
            print(final_results.iloc[-1:].to_string(header=False, index=False))


if __name__ == "__main__":
    # Windows에서 asyncio 이벤트 루프 정책 설정 (tqdm 관련)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
