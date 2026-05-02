# core/chatbot_base.py
from typing import List, Dict, Any, Optional, Union
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class RAGChatBot:
    """임베딩/Chroma/프롬프트/체인을 캡슐화. ask()로 호출."""
    def __init__(
        self,
        api_key: str,
        persist_directory: str,
        collection_name: str,
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = None,
        temperature: float = 0.0,
        request_timeout: int = 120,
        max_tokens: int = 3000,
        qa_system_prompt_text: Optional[str] = None,
        **kwargs, # 자식 클래스에서 추가 인자를 받을 수 있도록 허용
    ) -> None:
        # Embeddings & DB
        embedding_kwargs = {"api_key": api_key, "model": embedding_model}
        if embedding_dimensions:
            embedding_kwargs["dimensions"] = embedding_dimensions
        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        self.retriever = self.db.as_retriever()

        # LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=max_tokens,
            model_name=llm_model,
            request_timeout=request_timeout,
        )

        # Prompts
        contextualize_q_system = (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question, which can be understood "
            "without the chat history. Do NOT answer the question; just reformulate it."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        default_qa = (
            "당신은 무쉽따 pdf 내용에 대해 알려주는 챗봇입니다.\n"
            "**제공된 문맥만을 근거하여 답변하세요.**\n"
            "외부 지식이나 사전 학습 내용은 사용하지 마세요.\n"
            "문맥을 통해 알 수 없는 질문에는 모른다고 대답하세요.\n\n"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_text or default_qa),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "{context}"), # Explicitly add context as a system message
        ])

        # 문서 프롬프트도 외부에서 주입 가능하도록 수정
        doc_prompt_template = kwargs.pop("document_prompt_template", "[출처: {source}]\n{page_content}")
        self.document_prompt = PromptTemplate.from_template(doc_prompt_template)

        # Chains
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.question_answer_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=self.qa_prompt, document_prompt=self.document_prompt
        )
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

    @staticmethod
    def _to_msgs(history: Optional[List[Union[BaseMessage, Dict[str, str]]]]) -> List[BaseMessage]:
        if not history: return []
        out: List[BaseMessage] = []
        for t in history:
            if isinstance(t, BaseMessage):
                out.append(t)
            else:
                role, content = t.get("role"), t.get("content", "")
                out.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
        return out

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Union[BaseMessage, Dict[str, str]]]] = None,
        top_k: Optional[int] = None,
        keep_last: int = 10,
    ) -> Dict[str, Any]:
        if top_k is not None and hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = top_k

        # 실제 대화기록은 마지막 질문을 제외한 리스트
        history_for_chain = self._to_msgs(chat_history[:-1] if chat_history else [])

        out = self.rag_chain.invoke({
            "input": question,
            "chat_history": history_for_chain,
        })
        answer = out.get("answer", "")

        # 소스 정리
        sources = []
        for d in out.get("context", []) or []:
            md = getattr(d, "metadata", {}) or {}
            sources.append({"source": md.get("source"), "page": md.get("page")})

        # 프론트에서 받은 대화기록(질문 포함)에 AI 답변만 추가
        next_history = (chat_history or []) + [
            {"role": "ai", "content": answer},
        ]
        if keep_last: next_history = next_history[-keep_last:]
        return {"answer": answer, "sources": sources, "chat_history": next_history}