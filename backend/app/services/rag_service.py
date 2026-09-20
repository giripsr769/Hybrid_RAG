from app.retrieval.hybrid_retriever import HybridRetriever
from langchain_openai import ChatOpenAI
from app.guardrails.guardrail_service import (
    GuardrailService,
    GuardrailViolation
)

class RAGService:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        guardrail_service: GuardrailService
    ):
        self.hybrid_retriever = hybrid_retriever
        self.guardrail_service = guardrail_service

        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

        self.chat_history = []

    def build_context(
            self,
            hybrid_results: dict
        ) -> str:

            vector_documents = hybrid_results["vector_results"]
            graph_results = hybrid_results["graph_results"]

            # 1. Prepare Vector RAG context
            vector_context = "\n\n".join(
                document.page_content
                for document in vector_documents
            )

            # 2. Prepare Knowledge Graph context
            graph_context = "\n".join(
                f"{item['source']} --{item['relationship']}--> {item['target']}"
                for item in graph_results
            )

            # 3. Combine both contexts
            combined_context = f"""
                VECTOR CONTEXT:
                {vector_context}

                KNOWLEDGE GRAPH CONTEXT:
                {graph_context}
                """

            return combined_context

    def contextualize_question(
        self,
        question: str
    ) -> str:

        # No history means the question is already standalone
        if not self.chat_history:
            return question

        recent_history = self.chat_history[-10:]
        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_history
        )

        prompt = f"""
        You rewrite follow-up questions into standalone questions.

        Use the conversation history only to understand references
        such as "it", "its", "that", "this", "they", or similar terms.

        Do NOT answer the question.
        Do NOT add information that is not present in the conversation.
        If the question is already standalone, return it unchanged.

        CONVERSATION HISTORY:
        {history_text}

        CURRENT QUESTION:
        {question}

        STANDALONE QUESTION:
        """

        response = self.llm.invoke(prompt)

        return response.content.strip()

    def build_sources(
            self,
            hybrid_results: dict
        ) -> list[str]:

            sources = []

            for document in hybrid_results["vector_results"]:

                page_label = document.metadata.get(
                    "page_label"
                )

                if page_label and page_label not in sources:
                    sources.append(page_label)

            return sources

    def answer(
            self,
            question: str
        ) -> str:

            # 1. Validate user input before RAG processing
            self.guardrail_service.validate_input(
                question
            )

            # 1. Convert follow-up question into a standalone question
            standalone_question = self.contextualize_question(
                question
            )

            # 2. Retrieve evidence from FAISS + Neo4j
            hybrid_results = self.hybrid_retriever.retrieve(
                question=standalone_question,
                vector_k=4,
                graph_limit_per_entity=10
            )

            # 2. Build combined context
            context = self.build_context(
                hybrid_results
            )

            # 3. Create prompt for the final LLM
            prompt = f"""
                You are a helpful RAG assistant.

                Answer the user's question using ONLY the provided context.

                The context contains:
                1. VECTOR CONTEXT - relevant text retrieved from the uploaded PDF.
                2. KNOWLEDGE GRAPH CONTEXT - relationships extracted from the same PDF.

                Instructions:
                - Use both sources when they are relevant.
                - Do not invent information that is not supported by the context.
                - If the context does not contain enough information to answer,
                clearly say that the uploaded document does not provide enough information.
                - Give a clear and concise answer.

                CONTEXT:
                {context}

                USER QUESTION:
                {question}

                ANSWER:
                """

            # 4. Generate final answer
            response = self.llm.invoke(prompt)
            answer = response.content

            # Check whether the generated answer is grounded
            # in the retrieved RAG context
            grounding_result = self.guardrail_service.check_grounding(
                context=context,
                answer=answer
            )

            if not grounding_result.is_grounded:
                raise GuardrailViolation(
                    "The generated answer could not be verified against the document."
                )

            # Save current conversation to memory
            self.chat_history.append({
                "role": "user",
                "content": question
            })

            self.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            return answer

    def clear_history(self) -> None:
        self.chat_history.clear()

    def get_evaluation_data(
            self,
            question: str
        ) -> dict:

            standalone_question = self.contextualize_question(
                question
            )

            hybrid_results = self.hybrid_retriever.retrieve(
                question=standalone_question,
                vector_k=4,
                graph_limit_per_entity=10
            )

            context = self.build_context(
                hybrid_results
            )

            return {
                "question": standalone_question,
                "context": context
            }

    def generate_evaluation_sample(
            self,
            question: str
        ) -> dict:

            standalone_question = self.contextualize_question(
                question
            )

            hybrid_results = self.hybrid_retriever.retrieve(
                question=standalone_question,
                vector_k=4,
                graph_limit_per_entity=10
            )

            context = self.build_context(
                hybrid_results
            )

            prompt = f"""
        You are a helpful RAG assistant.

        Answer the user's question using ONLY the provided context.

        The context contains:
        1. VECTOR CONTEXT - relevant text retrieved from the uploaded PDF.
        2. KNOWLEDGE GRAPH CONTEXT - relationships extracted from the same PDF.

        Instructions:
        - Use both sources when they are relevant.
        - Do not invent information that is not supported by the context.
        - If the context does not contain enough information to answer,
        clearly say that the uploaded document does not provide enough information.
        - Give a clear and concise answer.

        CONTEXT:
        {context}

        USER QUESTION:
        {standalone_question}

        ANSWER:
        """

            response = self.llm.invoke(
                prompt
            )

            answer = response.content.strip()

            return {
                "question": standalone_question,
                "context": context,
                "answer": answer
            }