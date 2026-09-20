import os
import time

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph


class GraphBuilder:

    def __init__(self):

        # LLM used to identify entities and relationships
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

        # Converts normal text into graph structures
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm
        )

        # Connection to Neo4j
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    def test_connection(self) -> str:
        result = self.graph.query(
            """
            RETURN "Python connected to Neo4j Aura" AS message
            """
        )
        return result[0]["message"]

    def clear_graph(self):
        self.graph.query(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )
        

    def build(
        self,
        chunks: list[Document]
        ) -> int:


        if not chunks:
            raise ValueError(
                "Chunks cannot be empty."
            )

        print("1. Starting KG extraction...")
        start = time.time()

        graph_documents = (
            self.graph_transformer
            .convert_to_graph_documents(chunks)
        )

        print(
            f"2. KG extraction completed in "
            f"{time.time() - start:.2f} seconds"
        )

        print("3. Starting Neo4j write...")
        start = time.time()

        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

        print(
            f"4. Neo4j write completed in "
            f"{time.time() - start:.2f} seconds"
        )

        return len(graph_documents)