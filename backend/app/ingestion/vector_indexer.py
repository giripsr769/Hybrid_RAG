from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorIndexer:

    def __init__(
        self,
        index_path: str = "data/faiss_index"
    ):
        self.index_path = Path(index_path)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    def create_index(
        self,
        chunks: list[Document]
    ) -> FAISS:

        if not chunks:
            raise ValueError(
                "Chunks cannot be empty."
            )

        # Create embeddings and FAISS index
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Make sure directory exists
        self.index_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save FAISS index locally
        vector_store.save_local(
            str(self.index_path)
        )

        return vector_store