"""Embedding helper: text -> vector.

Two providers behind one tiny interface:
  * fastembed : REAL semantic model (BAAI/bge-small-en-v1.5), runs locally, no API key.
  * hash      : offline, deterministic stand-in used for tests / no-internet demos.
                It is lexical (word overlap), NOT semantic. Do not use it in production.
"""
import hashlib
import math
import re
from functools import lru_cache

from app.config import get_settings

# ---------------------------------------------------------------- hash provider
_STOP = set("""a an the and or of to in on for with is are was were be been do does did
i you we they he she it my our your their me us how what which who when where why can
could should would will may might have has had get gets got per this that these those
at by from as if than then so about into up out any all""".split())

# tiny demo shim so "vacation days" can find "annual leave" in offline mode.
# A real embedding model learns this kind of similarity by itself.
_ALIASES = {"vacation": "leave", "holiday": "leave", "holidays": "leave", "pto": "leave",
            "wfh": "remote", "homeoffice": "remote", "laptop": "equipment"}
_HASH_DIM = 512


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in _STOP:
            continue
        w = _ALIASES.get(w, w)
        if len(w) > 3 and w.endswith("s"):   # crude plural stemming
            w = w[:-1]
        out.append(w)
    return out


def _hash_embed(text: str) -> list[float]:
    toks = _tokens(text)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # unigrams + bigrams
    vec = [0.0] * _HASH_DIM
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)   # md5 = stable across runs
        idx = h % _HASH_DIM
        sign = 1.0 if (h >> 100) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ------------------------------------------------------------ fastembed provider
@lru_cache(maxsize=1)
def _fastembed_model():
    from fastembed import TextEmbedding  # imported lazily: heavy dependency
    return TextEmbedding(model_name=get_settings().embedding_model)


# ------------------------------------------------------------------ public API
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (used at ingestion time)."""
    provider = get_settings().embedding_provider
    if provider == "hash":
        return [_hash_embed(t) for t in texts]
    if provider == "fastembed":
        return [v.tolist() for v in _fastembed_model().embed(texts)]
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def embed_query(question: str) -> list[float]:
    """Embed a user question (used at query time).

    Uses the SAME model as the documents - vectors from different models
    live in different spaces and cannot be compared.
    """
    provider = get_settings().embedding_provider
    if provider == "hash":
        return _hash_embed(question)
    if provider == "fastembed":
        return next(iter(_fastembed_model().query_embed(question))).tolist()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def get_dimension() -> int:
    """Vector size. We MEASURE it from the model instead of hard-coding a number,
    because the Qdrant collection dimension must match the model exactly."""
    return len(embed_texts(["dimension probe"])[0])
