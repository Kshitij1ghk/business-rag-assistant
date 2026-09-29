from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import rag, vector_store
from app.models import AskRequest, AskResponse

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="Business RAG Assistant", version="0.1.0")


@app.get("/", include_in_schema=False)
def index():
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return {"message": "Business RAG Assistant API"}




@app.get("/api/documents", include_in_schema=False)
def list_documents():
    docs_dir = BASE_DIR / "data" / "documents"
    docs = []
    if docs_dir.exists():
        for p in sorted(docs_dir.iterdir()):
            if p.suffix.lower() in (".txt", ".md"):
                docs.append({
                    "name": p.name,
                    "size": p.stat().st_size,
                    "content": p.read_text(encoding="utf-8")
                })
    return docs


@app.get("/health")
def health():
    return {"status": "ok", "indexed": vector_store.collection_exists()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not vector_store.collection_exists():
        raise HTTPException(503, "No index found. Run `python -m app.ingest` first.")
    try:
        return rag.answer_question(req.question)
    except Exception as exc:  # LLM/network failure: don't leak internals to the client
        raise HTTPException(502, "The answer generator failed. Check server logs.") from exc
