from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """
    Responsible only for loading PDF files
    into LangChain Document objects.
    """

    def load(self, file_path: Path) -> list[Document]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        loader = PyPDFLoader(str(file_path))

        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable content found in the PDF."
            )

        return documents