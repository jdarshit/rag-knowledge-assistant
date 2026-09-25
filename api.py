from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.ingest import ingest_folder
from src.rag import ask
from src.schemas import AskRequest, AskResponse, DocumentInfo, DocumentsResponse
from src.vector_store import delete_document, list_documents

app = FastAPI(title="RAG Knowledge Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Simple check to confirm the API is running."""
    return {"status": "ok"}


@app.get("/documents", response_model=DocumentsResponse)
def get_documents() -> DocumentsResponse:
    """List every document currently stored, with its chunk count."""
    documents = list_documents()
    return DocumentsResponse(
        documents=[DocumentInfo(name=name, chunks=count) for name, count in documents.items()]
    )


@app.post("/ingest")
def run_ingest() -> dict:
    """Ingest every PDF currently in data/sample_pdfs."""
    results = ingest_folder("data/sample_pdfs")
    return {"ingested": results}


@app.delete("/documents/{name}")
def remove_document(name: str) -> dict:
    """Delete all chunks belonging to one file."""
    deleted = delete_document(name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{name}' found.")
    return {"deleted_chunks": deleted}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using the RAG pipeline."""
    history = [message.model_dump() for message in request.history]
    kwargs = {"history": history}
    if request.top_k is not None:
        kwargs["k"] = request.top_k
    if request.max_distance is not None:
        kwargs["max_distance"] = request.max_distance

    result = ask(request.question, **kwargs)
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        search_query=result.get("search_query"),
        closest_distance=result.get("closest_distance"),
    )
