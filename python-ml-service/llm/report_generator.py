"""
5주차: LLM 기반 자동 분석 리포트 생성
학원에서 배운 LangChain + FAISS + Ollama(Gemma) 구조 그대로 사용
→ LLM 역량 어필: RAG, 벡터DB, 임베딩, 프롬프트 엔지니어링
"""
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from typing import Optional


class SatelliteReportGenerator:
    def __init__(self, model_name: str = "gemma2"):
        # 임베딩 모델 (학원에서 사용한 것과 동일)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask"
        )

        # LLM (Ollama + Gemma)
        self.llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.3,
        )

        # 벡터 저장소 초기화
        self.vectorstore = None
        self.retriever = None

        # RAG 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_template("""
당신은 위성 영상 분석 전문가입니다.
아래 분석 데이터를 기반으로 한국어 리포트를 작성하세요.

분석 데이터:
{context}

질문: {question}

다음 형식으로 리포트를 작성하세요:
1. 분석 개요 (어떤 지역에서 어떤 변화가 감지되었는지)
2. 변화 상세 (변화율, 면적, 위치)
3. 추정 원인 (건축, 자연재해, 계절 변화 등)
4. 권장 조치 (추가 관측 필요 여부, 우선순위)

리포트:
""")

    def add_analysis_results(self, results: list[dict]):
        """분석 결과를 벡터DB에 저장"""
        texts = []
        for r in results:
            text = (
                f"지역: {r.get('region', '미지정')}, "
                f"좌표: ({r.get('x', 0)}, {r.get('y', 0)}), "
                f"변화율: {r.get('change_rate', 0):.1f}%, "
                f"변화 면적: {r.get('area_pixels', 0)}px, "
                f"분석 시간: {r.get('timestamp', 'N/A')}, "
                f"변화 유형: {r.get('change_type', '미분류')}"
            )
            texts.append(text)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        else:
            self.vectorstore.add_texts(texts)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

    def generate_report(self, question: str) -> str:
        """질문에 대한 분석 리포트 생성"""
        if self.retriever is None:
            return "분석 데이터가 없습니다. 먼저 이미지 분석을 실행하세요."

        # RAG 체인 구성 (학원에서 배운 구조)
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)


# 싱글톤 인스턴스
_generator: Optional[SatelliteReportGenerator] = None


def get_generator() -> SatelliteReportGenerator:
    global _generator
    if _generator is None:
        _generator = SatelliteReportGenerator()
    return _generator


if __name__ == "__main__":
    gen = SatelliteReportGenerator()

    # 테스트 데이터
    test_results = [
        {"region": "강남구 A구역", "x": 120, "y": 80,
         "change_rate": 15.3, "area_pixels": 2400,
         "timestamp": "2024-07-15", "change_type": "건축 활동"},
        {"region": "강남구 B구역", "x": 200, "y": 150,
         "change_rate": 8.7, "area_pixels": 1100,
         "timestamp": "2024-07-15", "change_type": "토지 변경"},
        {"region": "서초구 C구역", "x": 50, "y": 220,
         "change_rate": 3.2, "area_pixels": 500,
         "timestamp": "2024-07-15", "change_type": "식생 변화"},
    ]

    gen.add_analysis_results(test_results)

    print("리포트 생성 중... (Ollama가 실행 중이어야 합니다)")
    print("=" * 60)
    report = gen.generate_report("강남구 최근 변화를 분석해주세요")
    print(report)
