from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

import json
from pathlib import Path


class AnswerRelevanceResult(BaseModel):

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Answer relevance score from 0.0 to 1.0. "
            "1.0 means the answer directly addresses the question."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation for the relevance score."
        )
    )

class FaithfulnessResult(BaseModel):

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Faithfulness score from 0.0 to 1.0. "
            "1.0 means all factual claims in the answer "
            "are supported by the retrieved context."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation for the faithfulness score."
        )
    )

class RetrievalRelevanceResult(BaseModel):

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Retrieval relevance score from 0.0 to 1.0. "
            "1.0 means the retrieved context is highly relevant "
            "to the user's question."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation for the retrieval relevance score."
        )
    )

class RAGEvaluator:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

        self.answer_relevance_llm = (
            self.llm.with_structured_output(
                AnswerRelevanceResult
            )
        )
        self.faithfulness_llm = (
            self.llm.with_structured_output(
                FaithfulnessResult
            )
        )

        self.retrieval_relevance_llm = (
            self.llm.with_structured_output(
                RetrievalRelevanceResult
            )
        )
        

    def evaluate_answer_relevance(
            self,
            question: str,
            answer: str
        ) -> AnswerRelevanceResult:

            prompt = f"""
        You are an evaluator for a RAG application.

        Evaluate how relevant the generated answer is to the user's question.

        Scoring rules:
        - 1.0 = The answer directly and completely addresses the question.
        - 0.75 = The answer mostly addresses the question with minor irrelevant
        or missing information.
        - 0.5 = The answer partially addresses the question.
        - 0.25 = The answer is mostly irrelevant.
        - 0.0 = The answer does not address the question at all.

        Important:
        - Evaluate ONLY whether the answer addresses the user's question.
        - Do NOT evaluate factual correctness.
        - Do NOT evaluate completeness using your own knowledge.
        - Do NOT penalize the answer because additional facts could have been mentioned.
        - Do NOT use external knowledge.
        - Do NOT evaluate grounding or faithfulness.
        - If the answer directly responds to what the user asked, it should receive
        a high relevance score even if the answer is short.

        USER QUESTION:
        {question}

        GENERATED ANSWER:
        {answer}
        """

            result = self.answer_relevance_llm.invoke(
                prompt
            )
            print("evaluate_answer_relevance-Score: ",result)

            return result

    def evaluate_faithfulness(
            self,
            context: str,
            answer: str
        ) -> FaithfulnessResult:

            prompt = f"""
        You are a faithfulness evaluator for a RAG application.

        Evaluate whether the factual claims in the generated answer are
        supported by the retrieved context.

        Scoring rules:
        - 1.0 = All factual claims in the answer are supported by the context.
        - 0.75 = Most factual claims are supported, with a minor unsupported claim.
        - 0.5 = Some claims are supported and some are unsupported.
        - 0.25 = Most factual claims are unsupported.
        - 0.0 = The answer is completely unsupported or contradicts the context.

        Important:
        - Use ONLY the retrieved context.
        - Do NOT use your own external knowledge.
        - Do NOT evaluate whether the answer is relevant to the question.
        - Do NOT penalize the answer for being concise.
        - Evaluate only claims actually made in the answer.
        - If the answer contains multiple factual claims, consider how many of
        those claims are supported by the context.

        RETRIEVED CONTEXT:
        {context}

        GENERATED ANSWER:
        {answer}
        """

            result = self.faithfulness_llm.invoke(
                prompt
            )

            return result

    def evaluate_retrieval_relevance(
            self,
            question: str,
            context: str
        ) -> RetrievalRelevanceResult:

            prompt = f"""
        You are a retrieval relevance evaluator for a RAG application.

        Evaluate how relevant the retrieved context is for answering
        the user's question.

        Scoring rules:
        - 1.0 = The context contains highly relevant information that directly
        helps answer the question.
        - 0.75 = Most of the context is relevant, with some unnecessary information.
        - 0.5 = The context contains some useful information, but much of it
        is irrelevant or incomplete.
        - 0.25 = The context has very little useful information for the question.
        - 0.0 = The context is unrelated to the question.

        Important:
        - Evaluate ONLY the relationship between the question and retrieved context.
        - Do NOT evaluate the generated answer.
        - Do NOT use external knowledge.
        - Do NOT require every part of the context to be relevant.
        - Focus on whether the context provides useful evidence for answering
        the question.

        USER QUESTION:
        {question}

        RETRIEVED CONTEXT:
        {context}
        """

            result = self.retrieval_relevance_llm.invoke(
                prompt
            )

            return result

    def evaluate(
            self,
            question: str,
            context: str,
            answer: str
        ) -> dict:

            answer_relevance = (
                self.evaluate_answer_relevance(
                    question=question,
                    answer=answer
                )
            )

            faithfulness = (
                self.evaluate_faithfulness(
                    context=context,
                    answer=answer
                )
            )

            retrieval_relevance = (
                self.evaluate_retrieval_relevance(
                    question=question,
                    context=context
                )
            )

            return {
                "answer_relevance": answer_relevance,
                "faithfulness": faithfulness,
                "retrieval_relevance": retrieval_relevance
            }

    def load_evaluation_questions(
            self,
            file_path: str
        ) -> list[dict]:

            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Evaluation dataset not found: {file_path}"
                )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:
                questions = json.load(file)

            if not questions:
                raise ValueError(
                    "Evaluation dataset is empty."
                )

            return questions

    def run_evaluation(
            self,
            questions: list[dict],
            rag_service
        ) -> list[dict]:

            results = []

            for item in questions:

                question_id = item["id"]
                question = item["question"]
                answerable = item["answerable"]

                # Run the real Hybrid RAG pipeline
                sample = rag_service.generate_evaluation_sample(
                    question=question
                )

                # Evaluate the generated answer
                evaluation = self.evaluate(
                    question=sample["question"],
                    context=sample["context"],
                    answer=sample["answer"]
                )

                results.append({
                    "id": question_id,
                    "question": sample["question"],
                    "answerable": answerable,
                    "answer": sample["answer"],

                    "answer_relevance": {
                        "score": evaluation[
                            "answer_relevance"
                        ].score,
                        "reason": evaluation[
                            "answer_relevance"
                        ].reason
                    },

                    "faithfulness": {
                        "score": evaluation[
                            "faithfulness"
                        ].score,
                        "reason": evaluation[
                            "faithfulness"
                        ].reason
                    },

                    "retrieval_relevance": {
                        "score": evaluation[
                            "retrieval_relevance"
                        ].score,
                        "reason": evaluation[
                            "retrieval_relevance"
                        ].reason
                    }
                })

            return results
    
