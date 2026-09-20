from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)

from deepeval.test_case import LLMTestCase


class RAGEvaluator:

    def __init__(self):

        self.answer_relevancy = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-4.1-mini",
            include_reason=True
        )

        self.faithfulness = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4.1-mini",
            include_reason=True
        )

        self.contextual_relevancy = ContextualRelevancyMetric(
            threshold=0.7,
            model="gpt-4.1-mini",
            include_reason=True
        )

    def evaluate(
            self,
            question: str,
            context: str,
            answer: str
        ) -> dict:

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=[context]
            )

            # Answer Relevancy
            self.answer_relevancy.measure(
                test_case
            )

            # Faithfulness
            self.faithfulness.measure(
                test_case
            )

            # Contextual Relevancy
            self.contextual_relevancy.measure(
                test_case
            )

            return {
                "answer_relevancy": {
                    "score": self.answer_relevancy.score,
                    "reason": self.answer_relevancy.reason,
                    "passed": self.answer_relevancy.is_successful()
                },

                "faithfulness": {
                    "score": self.faithfulness.score,
                    "reason": self.faithfulness.reason,
                    "passed": self.faithfulness.is_successful()
                },

                "contextual_relevancy": {
                    "score": self.contextual_relevancy.score,
                    "reason": self.contextual_relevancy.reason,
                    "passed": self.contextual_relevancy.is_successful()
                }
            }