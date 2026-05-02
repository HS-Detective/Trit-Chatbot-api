# main.py
from core.chatbot_base import RAGChatBot
from core.settings import OPENAI_API_KEY
from core.utils import load_yaml, read_text

CFG = load_yaml("features/faq/config.yaml")
bot = RAGChatBot(
    api_key=OPENAI_API_KEY,
    persist_directory=CFG["paths"]["chroma_dir"],
    collection_name=CFG["collection_name"],
    llm_model=CFG["models"]["llm"],
    embedding_model=CFG["models"]["embedding"],
    qa_system_prompt_text=read_text(CFG["paths"]["system_prompt"]),
)

if __name__ == "__main__":
    chat_history = []
    print("콘솔 챗봇 시작 (exit 입력 시 종료)")
    while True:
        q = input("질문: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = bot.ask(q, chat_history=chat_history)
        a = out["answer"]
        print("답변:", a, "\n")
        chat_history = out["chat_history"][-10:]
