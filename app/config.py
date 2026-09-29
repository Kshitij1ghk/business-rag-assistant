"""Central configuration. Everything comes from environment variables / .env.

Rule: no secret ever appears in Python code or in Git.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real env vars still win

BASE_DIR = Path(__file__).resolve().parent.parent


# NOTE: a plain default like `x: str = os.getenv("X")` is evaluated ONCE at import time.
# default_factory runs on every Settings() call, so env changes (e.g. in tests) are seen.
def _str(name: str, default: str = ""):
    return field(default_factory=lambda: os.getenv(name, default))


def _int(name: str, default: int):
    return field(default_factory=lambda: int(os.getenv(name, default)))


def _float(name: str, default: float):
    return field(default_factory=lambda: float(os.getenv(name, default)))


@dataclass(frozen=True)
class Settings:
    # --- LLM (generation) ---
    llm_provider: str = _str("LLM_PROVIDER", "gemini")
    llm_api_key: str = _str("LLM_API_KEY", "")
    gemini_api_key: str = _str("GEMINI_API_KEY", "")
    llm_model: str = _str("LLM_MODEL", "gemini-3.5-flash-lite")

    # --- Embeddings (retrieval) ---
    embedding_provider: str = _str("EMBEDDING_PROVIDER", "fastembed")
    embedding_model: str = _str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Qdrant ---
    # If QDRANT_URL is empty we use an embedded, on-disk Qdrant (no server needed).
    qdrant_url: str = _str("QDRANT_URL", "")
    qdrant_api_key: str = _str("QDRANT_API_KEY", "")
    qdrant_path: str = _str("QDRANT_PATH", str(BASE_DIR / "qdrant_data"))
    qdrant_collection: str = _str("QDRANT_COLLECTION", "business_docs")

    # --- Documents + chunking + retrieval ---
    documents_dir: str = _str("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents"))
    chunk_size: int = _int("CHUNK_SIZE", 400)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 60)
    top_k: int = _int("TOP_K", 4)
    min_score: float = _float("MIN_SCORE", 0.40)


def get_settings() -> Settings:
    # Built fresh each call so tests can change env vars.
    return Settings()
