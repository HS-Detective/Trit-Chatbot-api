import pandas as pd
import argparse
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가하여 임포트 문제를 해결합니다.
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from routers.chat import BOTS

def run_evaluation(feature_id: str):
    """특정 기능(feature_id)에 대한 RAG 파이프라인의 성능을 평가합니다."""
    print(f"\n▶ '{feature_id}' 기능에 대한 RAG 성능 평가를 시작합니다...")

    # 1. 챗봇 및 평가 데이터셋 로드
    print("챗봇과 평가 데이터셋을 로드합니다...")
    try:
        chatbot = BOTS.get(feature_id)
        if not chatbot:
            print(f"[오류] '{feature_id}' 챗봇을 로드할 수 없습니다. 사용 가능한 기능: {list(BOTS.keys())}")
            return

        # 기능별 평가 데이터셋을 먼저 찾고, 없으면 공용 데이터셋을 사용합니다.
        eval_dataset_path = project_root / f"features/{feature_id}/eval_dataset.csv"
        if not eval_dataset_path.exists():
            eval_dataset_path = project_root / "evaluation/eval_dataset.csv"
            if not eval_dataset_path.exists():
                 print(f"[오류] '{feature_id}'에 대한 평가 데이터셋을 찾을 수 없습니다.")
                 return

        eval_df = pd.read_csv(eval_dataset_path)
        print(f"  - '{eval_dataset_path.name}'에서 {len(eval_df)}개의 평가 데이터를 로드했습니다.")

    except Exception as e:
        print(f"[오류] 평가 준비 중 오류 발생: {e}")
        return

    # 2. 데이터셋의 각 질문에 대해 챗봇 답변 및 컨텍스트 생성
    print("챗봇 답변 및 컨텍스트를 생성합니다...")
    results = []
    for index, row in eval_df.iterrows():
        question = row['question']
        print(f"  - 질문 처리 중 ({index+1}/{len(eval_df)}): {question[:50]}...")
        
        response = chatbot.rag_chain.invoke({"input": question, "chat_history": []})
        
        results.append({
            "question": question,
            "answer": response['answer'],
            "contexts": [doc.page_content for doc in response.get('context', [])],
            "ground_truth": row['ground_truth']
        })
    
    # 3. Ragas 평가 실행
    print("\nRagas로 평가를 실행합니다... (시간이 걸릴 수 있습니다)")
    dataset = Dataset.from_list(results)
    
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )

    # 4. 결과 출력
    print(f"\n📊 '{feature_id}' 기능 RAG 성능 평가 결과 📊")
    score_df = score.to_pandas()
    print(score_df.to_string())
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="챗봇의 RAG 성능을 평가합니다.")
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help=f"평가할 기능의 ID. 사용 가능: {list(BOTS.keys())}"
    )
    args = parser.parse_args()

    if args.feature not in BOTS:
        print(f"[오류] '{args.feature}'는 유효한 기능 ID가 아닙니다.")
        print(f"사용 가능한 기능: {list(BOTS.keys())}")
    else:
        run_evaluation(args.feature)
