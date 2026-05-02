# chatbot.py
from typing import List, Dict, Any, Optional, Union
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class RAGChatBot:
    """네 코드(임베딩/Chroma/프롬프트/체인)를 그대로 캡슐화.
    - 초기화 1회 → rag_chain 재사용
    - ask(question, chat_history) 호출만으로 응답
    """

    def __init__(
        self,
        api_key: str,
        persist_directory: str,
        collection_name: str,
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 0.0,
        request_timeout: int = 120,
        max_tokens: int = 3000,
        qa_system_prompt: Optional[str] = None,
    ) -> None:
        # 1) Embeddings & DB
        self.embeddings = OpenAIEmbeddings(api_key=api_key, model=embedding_model)
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        self.retriever = self.db.as_retriever()

        # 2) LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=max_tokens,
            model_name=llm_model,
            request_timeout=request_timeout
        )

        # 3) Prompts (네 코드 그대로)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question, which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if needed "
            "and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.qa_system_prompt = qa_system_prompt or (
            "당신은 무쉽따 pdf 내용에 대해 알려주는 챗봇입니다.\n"
            "**제공된 문맥만을 근거하여 답변하세요.**\n"
            "외부 지식이나 사전 학습 내용은 사용하지 마세요.\n"
            "문맥을 통해 알 수 없는 질문에는 모른다고 대답하세요.\n\n"
            "{context}\n"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        self.document_prompt = PromptTemplate.from_template(
            "[출처: {source}]\n{page_content}"
        )

        # 4) Chains
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.question_answer_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=self.qa_prompt, document_prompt=self.document_prompt
        )
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

    # --- utils ---
    @staticmethod
    def _to_msgs(history: Optional[List[Union[BaseMessage, Dict[str, str]]]]) -> List[BaseMessage]:
        if not history:
            return []
        msgs: List[BaseMessage] = []
        for t in history:
            if isinstance(t, BaseMessage):
                msgs.append(t)
            else:
                role = t.get("role")
                content = t.get("content", "")
                if role == "human":
                    msgs.append(HumanMessage(content=content))
                else:
                    msgs.append(AIMessage(content=content))
        return msgs

    # --- public ---
    def ask(
        self,
        question: str,
        chat_history: Optional[List[Union[BaseMessage, Dict[str, str]]]] = None,
        top_k: Optional[int] = None,
        keep_last: int = 10,
    ) -> Dict[str, Any]:
        if top_k is not None and hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = top_k

        history_msgs = self._to_msgs(chat_history)
        out = self.rag_chain.invoke({"input": question, "chat_history": history_msgs})
        answer = out.get("answer", "")

        sources = []
        for d in out.get("context", []) or []:
            md = getattr(d, "metadata", {}) or {}
            sources.append({
                "source": md.get("source"),
                "page": md.get("page"),
                "score": md.get("score"),
                "count": md.get("count"),
            })

        next_history = (chat_history or []) + [
            {"role": "human", "content": question},
            {"role": "ai", "content": answer},
        ]
        if keep_last:
            next_history = next_history[-keep_last:]

        return {"answer": answer, "sources": sources, "chat_history": next_history}
