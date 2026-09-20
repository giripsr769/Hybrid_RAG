from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 800, #default value in params
        chunk_overlap: int = 120 #default value in params
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split(
        self,
        documents: list[Document]
    ) -> list[Document]:

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        chunks = self.text_splitter.split_documents(
            documents
        )

        return chunks