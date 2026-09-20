from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.graph_retriever import GraphRetriever


class HybridRetriever:

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever


    def retrieve(
        self,
        question: str,
        vector_k: int = 4,
        graph_limit_per_entity: int = 10
    ) -> dict:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # Vector RAG
        vector_results = self.vector_retriever.retrieve(
            question=question,
            k=vector_k
        )

        # Knowledge Graph RAG
        graph_results = self.graph_retriever.retrieve(
            question=question,
            limit_per_entity=graph_limit_per_entity
        )

        return {
            "vector_results": vector_results,
            "graph_results": graph_results
        }