from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_chunker import TextChunker
from app.ingestion.vector_indexer import VectorIndexer
from app.ingestion.graph_builder import GraphBuilder
# from backend.app.evaluation.custom_rag_evaluator import RAGEvaluator


from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.rag_service import RAGService
from app.guardrails.guardrail_service import (
    GuardrailService,
    GuardrailViolation
)

from pathlib import Path
import shutil
from dotenv import load_dotenv
import os
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.evaluation.rag_evaluator import RAGEvaluator

# ---------------------------------
# Load Environment Variables
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# init Fastapi
app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    question: str


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

pdf_loader = PDFLoader()
vector_chunker = TextChunker(
    chunk_size=800,
    chunk_overlap=120
)

graph_chunker = TextChunker(
    chunk_size=3000,
    chunk_overlap=300
)
vector_indexer = VectorIndexer()
graph_builder = GraphBuilder()

vector_retriever = VectorRetriever()
graph_retriever = GraphRetriever()

guardrail_service = GuardrailService()

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    graph_retriever=graph_retriever
)
rag_service = RAGService(
    hybrid_retriever=hybrid_retriever,
    guardrail_service=guardrail_service
)

rag_evaluator = RAGEvaluator()



# -------------------------
# Health Check
# -------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Hybrid RAG API"
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_path = UPLOAD_DIR / file.filename

    # Save PDF
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

   # 2. Load PDF
    documents = pdf_loader.load(file_path)

    # 3. Split Documents
    vector_chunks = vector_chunker.split(documents)
    graph_chunks = graph_chunker.split(documents)

    # 4. Create FAISS Vector Index
    vector_indexer.create_index(vector_chunks)

    # 5. Clear OLD Knowledge Graph
    graph_builder.clear_graph()

    # 6. Create NEW Knowledge Graph
    graph_count = graph_builder.build(graph_chunks)

    # Clear conversation history for the previous document
    rag_service.clear_history()

    return {
    "status": "processed",
    "filename": file.filename,
    "pages": len(documents),
    "vector_chunks": len(vector_chunks),
    "graph_chunks": len(graph_chunks),
    "vector_index": "created",
    "graph_documents": graph_count,
    "knowledge_graph": "created"
}

@app.get("/test-neo4j")
def test_neo4j():
    graph_builder = GraphBuilder()
    message = graph_builder.test_connection()

    return {
        "status": "success",
        "message": message
    }

@app.get("/test-vector-search")
def test_vector_search(question: str):

    documents = vector_retriever.retrieve(
        question=question,
        k=4
    )

    results = []

    for document in documents:
        results.append({
            "content": document.page_content,
            "metadata": document.metadata
        })

    return {
        "question": question,
        "results_count": len(results),
        "results": results
    }


@app.get("/evaluate")
def evaluate_rag(
    question: str
    ):

        # Run the actual Hybrid RAG pipeline
        sample = rag_service.generate_evaluation_sample(
            question=question
        )

        # Evaluate using DeepEval
        evaluation = rag_evaluator.evaluate(
            question=sample["question"],
            context=sample["context"],
            answer=sample["answer"]
        )

        return {
            "question": sample["question"],
            "answer": sample["answer"],
            "evaluation": evaluation
        }
@app.post("/chat")
def chat(request: ChatRequest):

    try:
        answer = rag_service.answer(
            question=request.question
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except GuardrailViolation as error:
        raise HTTPException(
            status_code=400,
            detail=error.message
        )