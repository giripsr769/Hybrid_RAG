from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorRetriever:

    def __init__(
        self,
        index_path: str = "data/faiss_index"
    ):
        self.index_path = Path(index_path)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    def retrieve(
        self,
        question: str,
        k: int = 4
    ) -> list[Document]:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        # Load the FAISS index created during ingestion
        vector_store = FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Find the most relevant chunks
        documents = vector_store.similarity_search(
            query=question,
            k=k
        )

        return documents