# features/hs/chatbot.py
import json
import re
from typing import Dict, Any, Optional, List, Union

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from core.chatbot_base import RAGChatBot
from core.utils import read_text


# --- Pydantic 스키마 정의 ---

class PDResponse(BaseModel):
    """품목명 보강 단계의 출력 스키마"""
    item_description: str = Field(description="HS 해설서/법령체 스타일로 정제된 품목명 보강 설명", default="")
    missing_info: List[str] = Field(description="품목명 보강을 위해 필요한 정보 목록", default_factory=list)
    clarifying_questions: List[str] = Field(description="사용자 의도 명확화를 위한 질문 목록", default_factory=list)


class HSCandidate(BaseModel):
    """HS Code 후보 스키마"""
    hs_code: str = Field(description="추천 HS Code (4~10자리)")
    score: float = Field(description="후보의 적합도 점수 (0.0 ~ 1.0)")


class HSResponse(BaseModel):
    """HS Code 추론 단계의 출력 스키마"""
    hs_code: Optional[str] = Field(default=None, description="최종적으로 판단된 HS Code 10자리")
    reason: Optional[str] = Field(default=None, description="HS Code 판단 근거")
    need_info: Optional[str] = Field(default=None, description="추가로 필요한 정보나 사용자에게 할 구체적인 질문")
    hs_candidates: Optional[List[HSCandidate]] = Field(default=None, description="가능성 있는 HS Code 후보 목록")


class HSChatBot(RAGChatBot):
    """
    HS Code 상담을 위한 특화 챗봇.
    - 1단계: 사용자 질문을 바탕으로 품목명을 명확하게 보강하거나, 추가 정보를 요청합니다.
    - 2단계: 보강된 품목명으로 1차 검색 및 추론을 통해 유력한 HS Code 4자리를 특정하거나, 추가 정보를 요청합니다.
    - 3단계: 특정된 HS Code 4자리에 해당하는 모든 문서를 DB에서 가져와 최종 답변을 생성합니다.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        llm_params = kwargs.get("llm_params", {})
        pd_temperature = llm_params.get("pd_temperature", 0.1)
        hs_temperature = llm_params.get("hs_temperature", 0.0)
        max_tokens = llm_params.get("max_tokens", 3000)
        request_timeout = llm_params.get("request_timeout", 500)

        # 1. 품목명 보강(PD)을 위한 LLM 설정
        self.pd_llm = ChatOpenAI(
            temperature=pd_temperature,
            max_tokens=max_tokens,
            model_name=self.llm.model_name,
            request_timeout=request_timeout,
            openai_api_key=self.llm.openai_api_key,
        )

        # 2. HS Code 추론을 위한 LLM 재설정
        self.hs_llm = ChatOpenAI(
            temperature=hs_temperature,
            max_tokens=max_tokens,
            model_name=self.llm.model_name,
            request_timeout=request_timeout,
            openai_api_key=self.llm.openai_api_key,
        )

        # 3. 서비스용/평가용 체인 분리 생성
        pd_parser = PydanticOutputParser(pydantic_object=PDResponse)
        hs_parser = PydanticOutputParser(pydantic_object=HSResponse)
        pd_user_prompt_text = read_text(kwargs["paths"]["pd_user_prompt"])
        human_message_template = "질문: {input}\n\n관련 문서 발췌:\n{context}"

        # --- 서비스용(prod) 체인 생성 ---
        pd_system_prod_text = read_text(kwargs["paths"]["pd_system_prompt_prod"])
        hs_system_prod_text = read_text(kwargs["paths"]["system_prompt_prod"])
        
        pd_prompt_prod = ChatPromptTemplate.from_messages(
            [('system', pd_system_prod_text), ('human', pd_user_prompt_text)]
        ).partial(format_instructions=pd_parser.get_format_instructions())
        self.pd_chain_prod = pd_prompt_prod | self.pd_llm | pd_parser

        hs_qa_prompt_prod = ChatPromptTemplate.from_messages(
            [('system', hs_system_prod_text), ('human', human_message_template)]
        ).partial(format_instructions=hs_parser.get_format_instructions())
        self.question_answer_chain_prod = ({"context": lambda x: x["context"], "input": lambda x: x["input"]} 
                                         | hs_qa_prompt_prod | self.hs_llm | hs_parser)

        # --- 평가용(eval) 체인 생성 ---
        pd_system_eval_text = read_text(kwargs["paths"]["pd_system_prompt_eval"])
        hs_system_eval_text = read_text(kwargs["paths"]["system_prompt_eval"])

        pd_prompt_eval = ChatPromptTemplate.from_messages(
            [('system', pd_system_eval_text), ('human', pd_user_prompt_text)]
        ).partial(format_instructions=pd_parser.get_format_instructions())
        self.pd_chain_eval = pd_prompt_eval | self.pd_llm | pd_parser

        hs_qa_prompt_eval = ChatPromptTemplate.from_messages(
            [('system', hs_system_eval_text), ('human', human_message_template)]
        ).partial(format_instructions=hs_parser.get_format_instructions())
        self.question_answer_chain_eval = ({"context": lambda x: x["context"], "input": lambda x: x["input"]} 
                                          | hs_qa_prompt_eval | self.hs_llm | hs_parser)
        
        # 4. 대화기록을 단일 질문으로 압축하는 체인 추가
        condense_question_prompt_text = """이전 대화 내용과 새로운 질문이 주어졌을 때, 새로운 질문을 문맥을 포함하는 완전한 단일 질문으로 다시 작성해 주세요.

대화 내용:
{chat_history}

새로운 질문: {question}

단일 질문:"""
        condense_question_prompt = PromptTemplate.from_template(condense_question_prompt_text)
        self.standalone_question_chain = condense_question_prompt | self.llm | StrOutputParser()

        # --- Retriever 재정의 ---
        # RAGChatBot에서 생성된 기본 retriever가 config의 임베딩 차원을 존중하도록 재정의합니다.
        print(f'임베딩 차원({kwargs.get("models", {}).get("embedding_dimensions")})에 맞게 Retriever를 재정의합니다...')
        
        emb_model_name = kwargs.get("embedding_model")
        emb_dims = kwargs.get("embedding_dimensions")
        api_key = kwargs.get("api_key")
        persist_directory = kwargs.get("persist_directory")
        collection_name = kwargs.get("collection_name")
        search_type = kwargs.get("retriever", {}).get("search_type", "similarity")
        k = kwargs.get("retriever", {}).get("k", 5)

        embedding_function = OpenAIEmbeddings(
            model=emb_model_name,
            dimensions=emb_dims,
            openai_api_key=api_key
        )

        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )

        self.retriever = self.db.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k}
        )
        print("Retriever 재정의 완료.")

    @staticmethod
    def _format_hs_code(code: str) -> str:
        """HS Code 길이에 따라 서식을 적용합니다."""
        if not code:
            return ""
        # 숫자만 추출하여 정리
        cleaned_code = re.sub(r'[^0-9]', '', str(code))

        if len(cleaned_code) == 10:
            return f"{cleaned_code[:4]}.{cleaned_code[4:6]}-{cleaned_code[6:]}"
        elif len(cleaned_code) == 6:
            return f"{cleaned_code[:4]}.{cleaned_code[4:]}"
        else:
            return cleaned_code # 6자리, 10자리가 아니면 정리된 숫자 코드 반환

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Union[BaseMessage, Dict[str, str]]]] = None,
        top_k: Optional[int] = None,
        keep_last: int = 10,
        evaluation_mode: bool = False,
    ) -> Dict[str, Any]:
        print("\n--- HS Chatbot Invoked ---")

        if evaluation_mode:
            pd_chain = self.pd_chain_eval
            question_answer_chain = self.question_answer_chain_eval
            print("[INFO] Running in Evaluation Mode.")
        else:
            pd_chain = self.pd_chain_prod
            question_answer_chain = self.question_answer_chain_prod
            print("[INFO] Running in Production Mode.")
        
        final_question = question
        if chat_history:
            print("[INFO] 대화 기록을 사용하여 단일 질문 생성 중...")
            history_messages = []
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content")
                if role == 'human':
                    history_messages.append(HumanMessage(content=content))
                elif role == 'ai':
                    history_messages.append(AIMessage(content=content))

            final_question = self.standalone_question_chain.invoke({
                "chat_history": history_messages,
                "question": question,
            })
            print(f"[INFO] 생성된 단일 질문: {final_question}")

        try:
            # 1. 품목명 보강
            print(f'[1/8] 품목명 보강 시작: "{final_question}"')
            pd_result: PDResponse = pd_chain.invoke({"question": final_question})
            print(f"[2/8] 품목명 보강 결과: {json.dumps(pd_result.model_dump(), indent=2, ensure_ascii=False)}")

            # 1-1. 품목명 보강 단계에서 추가 정보가 필요한 경우
            if pd_result.clarifying_questions and not evaluation_mode:
                combined_question = "품목을 더 명확히 알기 위해 몇 가지 질문이 있습니다:\n" + "\n".join(f"- {q}" for q in pd_result.clarifying_questions)
                print("[INFO] 품목명 보강을 위해 추가 정보 필요. 사용자에게 질문합니다.")
                return self._format_output(
                    answer={"need_info": combined_question},
                    context=[],
                    question=question,
                    chat_history=chat_history,
                    keep_last=keep_last
                )

            pd_description = pd_result.item_description
            if not pd_description.strip():
                print("[INFO] 보강된 품목명이 없습니다. 원본 질문을 검색에 사용합니다.")
                pd_description = final_question

            # 2. 1차 검색 및 추론
            print("[3/8] 1차 문서 검색 시작...")
            retrieved_docs_1st = self.retriever.invoke(pd_description)
            print(f"[4/8] 1차 문서 검색 완료: {len(retrieved_docs_1st)}개 문서 발견")

            initial_hs_result: HSResponse = question_answer_chain.invoke(
                {"context": retrieved_docs_1st, "input": final_question}
            )
            print(f"[5/8] 1차 추론 답변: {json.dumps(initial_hs_result.model_dump(), indent=2, ensure_ascii=False)}")

            # 2-1. 1차 추론에서 추가 정보가 필요한 경우
            if initial_hs_result.need_info and not evaluation_mode:
                print("[INFO] 1차 추론에서 추가 정보 필요. 사용자에게 질문합니다.")
                return self._format_output(
                    answer=initial_hs_result.model_dump(),
                    context=retrieved_docs_1st,
                    question=question,
                    chat_history=chat_history,
                    keep_last=keep_last,
                )

            # 2-2. HS Code 후보 추출
            hs4_code = None
            if initial_hs_result.hs_code:
                hs4_code = self._format_hs_code(initial_hs_result.hs_code)[:4]
            elif initial_hs_result.hs_candidates:
                top_candidate = max(initial_hs_result.hs_candidates, key=lambda c: c.score, default=None)
                if top_candidate:
                    hs4_code = self._format_hs_code(top_candidate.hs_code)[:4]

            if not hs4_code:
                print("[WARN] 1차 추론에서 HS Code를 찾지 못했습니다. 1차 추론 결과를 반환합니다.")
                return self._format_output(initial_hs_result.model_dump(), retrieved_docs_1st, question, chat_history, keep_last)

            print(f"[6/8] HS Code 4자리 추출 완료: {hs4_code}")

            # 3. 2차 검색 및 최종 답변 생성
            print(f"[7/8] {hs4_code} 관련 전체 문서 검색 시작...")
            retrieved_docs_2nd = self._get_all_docs_by_hs4(hs4_code)
            if not retrieved_docs_2nd:
                print(f"[WARN] {hs4_code}에 해당하는 문서를 찾지 못했습니다. 1차 답변을 반환합니다.")
                return self._format_output(initial_hs_result.model_dump(), retrieved_docs_1st, question, chat_history, keep_last)
            print(f"[7/8] 전체 문서 검색 완료: {len(retrieved_docs_2nd)}개 문서 발견")
            print("---\nretrieved_docs_2nd content ---")
            for i, doc in enumerate(retrieved_docs_2nd):
                print(f"Document {i+1}:\n{doc.page_content}\n---")
            print("--- End of retrieved_docs_2nd content ---")

            print("[8/8] 최종 답변 생성 시작...")
            final_hs_result: HSResponse = question_answer_chain.invoke(
                {"context": retrieved_docs_2nd, "input": final_question}
            )
            print(f"--- HS Chatbot Finished ---\n{json.dumps(final_hs_result.model_dump(), indent=2, ensure_ascii=False)}")
            return self._format_output(final_hs_result.model_dump(), retrieved_docs_2nd, question, chat_history, keep_last)

        except Exception as e:
            import traceback
            print(f"[FATAL] HSChatBot.ask 실행 중 심각한 오류 발생: {e}")
            traceback.print_exc()
            error_answer = f"요청을 처리하는 중 심각한 오류가 발생했습니다: {e}"
            return self._format_output(error_answer, [], question, chat_history, keep_last)

    def _get_all_docs_by_hs4(self, hs4_code: str) -> List[Document]:
        """ChromaDB에서 특정 HS_4 메타데이터를 가진 모든 문서를 가져옵니다."""
        try:
            where_filter = {"HS_4": hs4_code}
            results = self.db._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
                limit=1000,
            )
            return [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results.get("documents", []), results.get("metadatas", []))
            ]
        except Exception as e:
            print(f"[ERROR] HS4 코드로 문서 검색 중 오류: {e}")
            return []

    def _format_output(
        self,
        answer: Union[Dict[str, Any], str],
        context: List[Document],
        question: str,
        chat_history: Optional[List[Dict[str, str]]],
        keep_last: int,
    ) -> Dict[str, Any]:
        """최종 응답 형식을 만듭니다."""
        sources = []
        if context:
            for d in context:
                md = getattr(d, "metadata", {}) or {}
                sources.append(
                    {
                        "source": md.get("source"),
                        "HS_4": md.get("HS_4"),
                        "HS_2": md.get("HS_2"),
                        "data": md.get("data"),
                    }
                )

        response_content = ""
        history_content = ""

        if isinstance(answer, dict):
            if "need_info" in answer and answer["need_info"]:
                response_content = answer["need_info"]
                history_content = response_content
            else:
                # 최종 답변을 문장 형태로 재구성
                hs_code = answer.get("hs_code")
                reason = answer.get("reason")
                
                formatted_code = self._format_hs_code(hs_code) if hs_code else "알 수 없음"
                
                if reason:
                    sentence = f"요청하신 품목의 HS Code는 {formatted_code}으로 판단됩니다.\n\n[판단 근거]\n{reason}"
                else:
                    sentence = f"요청하신 품목의 HS Code는 {formatted_code}으로 판단됩니다."

                response_content = sentence
                history_content = sentence
        else:
            response_content = str(answer)
            history_content = str(answer)

        next_history = (chat_history or []) + [
            {"role": "human", "content": question},
            {"role": "ai", "content": history_content},
        ]
        if keep_last:
            next_history = next_history[-keep_last:]

        return {"answer": response_content, "sources": sources, "chat_history": next_history}
