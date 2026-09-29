"""Ingestion pipeline: documents -> chunks -> embeddings -> Qdrant.

Run ONCE (or whenever documents change), separately from question answering:
    python -m app.ingest
"""
import re
from dataclasses import dataclass
from pathlib import Path

from app import embeddings, vector_store
from app.config import get_settings

SUPPORTED = {".txt", ".md"}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def load_documents(directory: str) -> dict[str, str]:
    """Read every .txt / .md file. Returns {filename: text}."""
    docs = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in SUPPORTED:
            docs[path.name] = path.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole lines into chunks of at most `chunk_size` characters.

    * Lines are kept whole, so a sentence about '20 days of annual leave' is not cut in half.
    * The next chunk starts with the last ~`overlap` characters of the previous one,
      so information sitting on a boundary is not lost.
    * A single very long line is split on sentence ends.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    units = []
    for ln in lines:
        if len(ln) <= chunk_size:
            units.append(ln)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if s.strip())

    chunks, current = [], ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail   # start on a word boundary
            current = f"{tail} {unit}".strip()
        else:
            current = f"{current}\n{unit}".strip() if current else unit
    if current:
        chunks.append(current)
    return chunks


def build_chunks(docs: dict[str, str], chunk_size: int, overlap: int) -> list[Chunk]:
    out = []
    for name, text in docs.items():
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append(Chunk(text=piece, source=name, chunk_index=i))
    return out


def ingest(recreate: bool = True) -> int:
    s = get_settings()
    docs = load_documents(s.documents_dir)
    if not docs:
        raise SystemExit(f"No .txt/.md documents found in {s.documents_dir}")
    chunks = build_chunks(docs, s.chunk_size, s.chunk_overlap)
    vectors = embeddings.embed_texts([c.text for c in chunks])
    vector_store.ensure_collection(dimension=len(vectors[0]), recreate=recreate)
    n = vector_store.upsert_chunks(chunks, vectors)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks -> indexed {n} chunks "
          f"(dimension={len(vectors[0])}, provider={s.embedding_provider})")
    return n


if __name__ == "__main__":
    ingest()
