# Hybrid RAG Application

A full-stack **Hybrid Retrieval-Augmented Generation (RAG)** application
that combines **FAISS vector retrieval** and a **Neo4j Knowledge Graph**
to answer questions from an uploaded PDF.

The application uses a **FastAPI backend**, **Streamlit frontend**,
**OpenAI models**, **LangChain**, guardrails, conversational memory, and
**DeepEval** for RAG evaluation.

## Architecture

``` text
User
  |
  v
Streamlit Frontend
  |
  v
FastAPI Backend
  |
  +-------------------+
  |                   |
  v                   v
FAISS Vector RAG   Neo4j Graph RAG
  |                   |
  +---------+---------+
            |
            v
      Hybrid Retriever
            |
            v
       RAG Context
            |
            v
        OpenAI LLM
            |
            v
       Guardrails
            |
            v
        Final Answer
```

During PDF ingestion, the document is split separately for vector
indexing and knowledge-graph extraction.

## Features

-   Upload and process a PDF
-   PDF text extraction with PyPDF/LangChain
-   Separate chunking strategies for Vector RAG and Graph RAG
-   OpenAI embeddings
-   FAISS vector index
-   Neo4j Aura Knowledge Graph
-   LLM-based entity extraction
-   Hybrid FAISS + Neo4j retrieval
-   Grounded answer generation
-   Conversational follow-up memory
-   Prompt-injection input guardrail
-   Answer-grounding output guardrail
-   DeepEval RAG evaluation
-   FastAPI REST API
-   Streamlit user interface
-   Dockerized frontend and backend
-   Docker Compose support

## Technology Stack

  Layer              Technology
  ------------------ ------------------------
  Frontend           Streamlit
  Backend            FastAPI, Uvicorn
  RAG Framework      LangChain
  Vector Database    FAISS
  Knowledge Graph    Neo4j Aura
  Embeddings         OpenAI Embeddings
  LLM                OpenAI
  PDF Processing     PyPDF / LangChain
  Evaluation         DeepEval
  Containerization   Docker, Docker Compose

## Project Structure

``` text
Hybrid_RAG/
|
+-- backend/
|   +-- app/
|   |   +-- evaluation/
|   |   |   +-- custom_rag_evaluator.py
|   |   |   +-- rag_evaluator.py
|   |   +-- guardrails/
|   |   |   +-- guardrail_service.py
|   |   +-- ingestion/
|   |   |   +-- graph_builder.py
|   |   |   +-- pdf_loader.py
|   |   |   +-- text_chunker.py
|   |   |   +-- vector_indexer.py
|   |   +-- retrieval/
|   |   |   +-- graph_retriever.py
|   |   |   +-- hybrid_retriever.py
|   |   |   +-- vector_retriever.py
|   |   +-- services/
|   |       +-- rag_service.py
|   +-- data/
|   |   +-- faiss_index/
|   |   +-- uploads/
|   +-- Dockerfile
|   +-- main.py
|
+-- frontend/
|   +-- app.py
|   +-- Dockerfile
|
+-- .env
+-- .gitignore
+-- docker-compose.yml
+-- requirements.txt
+-- README.md
```

`venv/`, cache directories, local secrets, uploaded documents, and
generated indexes should not be committed to Git.

## RAG Workflow

### 1. PDF Ingestion

The uploaded PDF is:

1.  Saved by the backend.
2.  Loaded using the PDF loader.
3.  Split into vector chunks.
4.  Split separately into larger graph chunks.
5.  Embedded and stored in FAISS.
6.  Processed by the graph transformer.
7.  Written to Neo4j Aura.

### 2. Query Processing

When the user asks a question:

1.  Input guardrails validate the question.
2.  Chat history is used to rewrite follow-up questions into standalone
    questions.
3.  FAISS retrieves semantically relevant document chunks.
4.  Neo4j retrieves relevant graph relationships.
5.  The results are combined into hybrid RAG context.
6.  The LLM generates an answer using the retrieved context.
7.  The output guardrail verifies that the answer is grounded.
8.  The answer is returned to the frontend.

## DeepEval

The project uses DeepEval to measure RAG quality.

Current metrics include:

-   **Answer Relevancy** --- whether the answer addresses the user's
    question.
-   **Faithfulness** --- whether the generated answer is supported by
    retrieved context.
-   **Contextual Relevancy** --- whether the retrieved context itself is
    relevant to the question.

The evaluation endpoint is intended primarily for development/testing
because evaluation introduces additional LLM calls, latency, and cost.

Example:

``` text
GET /evaluate?question=What%20does%20Next.js%20use?
```

## Environment Variables

Create a `.env` file in the project root:

``` env
OPENAI_API_KEY=your_openai_api_key

NEO4J_URI=your_neo4j_aura_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

Never commit `.env` to GitHub.

## Run Locally Without Docker

### 1. Create a virtual environment

Windows:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

``` powershell
pip install -r requirements.txt
```

### 3. Start FastAPI

From the `backend` directory:

``` powershell
uvicorn main:app --reload --port 8000
```

FastAPI:

``` text
http://127.0.0.1:8000
```

Swagger:

``` text
http://127.0.0.1:8000/docs
```

### 4. Start Streamlit

From the project root:

``` powershell
streamlit run frontend/app.py
```

Frontend:

``` text
http://127.0.0.1:8501
```

## Docker

The project uses separate Docker containers for the frontend and
backend.

### Backend

The backend container runs:

``` text
FastAPI / Uvicorn
Port: 8000
```

### Frontend

The frontend container runs:

``` text
Streamlit
Port: 8501
```

Within Docker Compose, Streamlit communicates with FastAPI using the
Docker service hostname:

``` text
http://backend:8000
```

The frontend should read this from the `BACKEND_URL` environment
variable rather than hard-coding localhost.

## Run with Docker Compose

Make sure Docker Desktop is running, then from the project root:

``` powershell
docker compose up --build
```

After startup:

``` text
Frontend:
http://localhost:8501

FastAPI Swagger:
http://localhost:8000/docs
```

To stop the containers:

``` powershell
docker compose down
```

To inspect running containers:

``` powershell
docker compose ps
```

To view logs:

``` powershell
docker compose logs -f
```

## Docker Data Persistence

The backend data directory is mounted as a Docker volume/bind mount so
generated FAISS indexes and uploaded files can survive container
recreation when configured in `docker-compose.yml`.

Example:

``` yaml
volumes:
  - ./backend/data:/app/data
```

## Deployment Plan

The intended deployment pipeline is:

``` text
Docker
  |
  v
GitHub
  |
  v
Hostinger VPS
  |
  v
Docker Compose
  |
  v
Nginx Reverse Proxy
  |
  v
SSL / HTTPS
  |
  v
Domain
```

On the VPS, the application will run in Docker containers. Nginx will
act as the public reverse proxy, and SSL will provide HTTPS access.

## Security Notes

-   Never commit `.env`.
-   Never commit API keys or Neo4j credentials.
-   Do not load untrusted FAISS pickle/index files.
-   Keep the FastAPI backend private behind the reverse proxy when
    deploying publicly.
-   Restrict public VPS firewall ports to those actually required.
-   Keep Docker images and Python dependencies updated.
-   Uploaded documents should be validated and size-limited before a
    production release.

## Current Status

The core Hybrid RAG pipeline is implemented:

-   PDF ingestion
-   Vector RAG
-   Knowledge Graph RAG
-   Hybrid retrieval
-   Chat memory
-   Input guardrails
-   Output grounding guardrail
-   DeepEval integration
-   FastAPI backend
-   Streamlit frontend
-   Docker configuration

Planned production improvements include retrieval optimization,
citations/source-page display, stronger multi-user/session isolation,
document-scoped graph management, background ingestion, caching, and
production deployment hardening.

## License

This project is currently intended for learning, experimentation, and
portfolio/buildathon use.
