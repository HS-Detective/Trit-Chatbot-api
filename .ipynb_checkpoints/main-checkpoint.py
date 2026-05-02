# main.py
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import RAGChatBot
from settings import (
    OPENAI_API_KEY, CHROMA_DIR, COLLECTION_NAME_FEATURE1,
    LLM_MODEL_FEATURE1, EMBED_MODEL_FEATURE1
)

bot = RAGChatBot(
    api_key=OPENAI_API_KEY,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME_FEATURE1,
    llm_model=LLM_MODEL_FEATURE1,
    embedding_model=EMBED_MODEL_FEATURE1
)

def run_console():
    chat_history = []
    print("콘솔 챗봇 시작 (exit 입력 시 종료)")
    while True:
        q = input("질문: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = bot.ask(q, chat_history=chat_history)
        a = out["answer"]
        print("답변:", a, "\n")
        # 히스토리 갱신 (객체로 유지해도 되고 dict로 유지해도 됩니다)
        chat_history.extend([{"role":"human","content":q},{"role":"ai","content":a}])
        chat_history = chat_history[-10:]

if __name__ == "__main__":
    run_console()
