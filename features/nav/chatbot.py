from typing import Dict, Any, Optional, List, Union

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import BaseMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from core.chatbot_base import RAGChatBot

class NavChatBot(RAGChatBot):
    """
    메뉴 안내를 위한 특화 챗봇.
    - MultiQueryRetriever를 사용하여 검색 성능 향상
    - '표/그래프/지도 설명'과 같은 후속 질문 처리 기능 추가
    """
    def __init__(self, **kwargs):
        retriever_k = kwargs.pop('retriever_k')
        super().__init__(**kwargs)
        
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.db.as_retriever(search_kwargs={"k": retriever_k}),
            llm=self.llm
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )
        
        self.last_doc_context = None

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Union[BaseMessage, Dict[str, str]]]] = None,
        top_k: Optional[int] = None,
        keep_last: int = 10,
    ) -> Dict[str, Any]:
        
        if self.last_doc_context:
            metadata = self.last_doc_context[0].metadata
            answer = None
            if "표" in question and "설명" in question:
                answer = metadata.get("table_info") or "관련 표 설명이 없습니다."
            elif "그래프" in question and "설명" in question:
                answer = metadata.get("graph_info") or "관련 그래프 설명이 없습니다."
            elif "지도" in question and "설명" in question:
                answer = metadata.get("map_info") or "관련 지도 설명이 없습니다."
            
            if answer:
                next_history = (chat_history or []) + [
                    {"role": "human", "content": question},
                    {"role": "ai", "content": answer},
                ]
                return {"answer": answer, "sources": [], "chat_history": next_history[-keep_last:]}

        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            if hasattr(self.retriever.retriever, "search_kwargs"):
                self.retriever.retriever.search_kwargs["k"] = top_k

        history_msgs = self._to_msgs(chat_history)
        rag_output = self.rag_chain.invoke({"input": question, "chat_history": history_msgs})
        
        answer = rag_output.get("answer", "")
        self.last_doc_context = rag_output.get("context", [])

        sources = []
        for doc in self.last_doc_context:
            md = getattr(doc, "metadata", {}) or {}
            sources.append({
                "source": md.get("source"),
                "menu_path": md.get("menu_path"),
                "url": md.get("url"),
                "login_required": md.get("login_required"),
            })

        next_history = (chat_history or []) + [
            {"role": "human", "content": question},
            {"role": "ai", "content": answer},
        ]
        
        return {"answer": answer, "sources": sources, "chat_history": next_history[-keep_last:]}