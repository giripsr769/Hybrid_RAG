import os
import json

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class EntityExtraction(BaseModel):

    entities: list[str] = Field(
        description=(
            "Important named entities, technologies, services, "
            "databases, components, or concepts from the question."
        )
    )


class GraphRetriever:

    def __init__(self):

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )


    def extract_entities(
            self,
            question: str
        ) -> list[str]:

            if not question.strip():
                raise ValueError(
                    "Question cannot be empty."
                )

            # Force the LLM to return our EntityExtraction structure
            structured_llm = self.llm.with_structured_output(
                EntityExtraction
            )

            prompt = f"""
        You are an entity extraction assistant for a Knowledge Graph.

        Extract important named entities, technologies, services,
        databases, components, or concepts from the user's question.

        Examples:

        Question: What does Next.js use?
        Entities: ["Next.js"]

        Question: How is React used?
        Entities: ["React"]

        Question: What does PostgreSQL store?
        Entities: ["PostgreSQL"]

        Question: What does the Playback Service manage?
        Entities: ["Playback Service"]

        Question: What database is used for analytics?
        Entities: ["Database", "Analytics"]

        Question: What technologies are used in the frontend?
        Entities: ["Technology", "Frontend"]

        User question:
        {question}
        """

            result = structured_llm.invoke(prompt)
            return result.entities

    def retrieve_by_entity(
            self,
            entity_name: str,
            limit: int = 10
        ) -> list[dict]:

            if not entity_name.strip():
                raise ValueError(
                    "Entity name cannot be empty."
                )

            query = """
            MATCH (a)-[r]-(b)
            WHERE NOT a:Document
            AND NOT b:Document
            AND replace(
                    replace(toLower(a.id), '.', ''),
                    ' ',
                    ''
                ) =
                replace(
                    replace(toLower($entity_name), '.', ''),
                    ' ',
                    ''
                )

            RETURN
                a.id AS source,
                type(r) AS relationship,
                b.id AS target
            LIMIT $limit
            """

            results = self.graph.query(
                query,
                params={
                    "entity_name": entity_name,
                    "limit": limit
                }
            )
            return results


    def retrieve(
            self,
            question: str,
            limit_per_entity: int = 10
        ) -> list[dict]:

            entities = self.extract_entities(question)

            graph_results = []

            for entity in entities:

                results = self.retrieve_by_entity(
                    entity_name=entity,
                    limit=limit_per_entity
                )

                graph_results.extend(results)

            return graph_results
