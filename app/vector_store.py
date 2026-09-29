"""All Qdrant operations live here. Nothing else in the app imports qdrant_client."""
import atexit
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:                                   # Qdrant server / Qdrant Cloud
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
    else:
        client = QdrantClient(path=s.qdrant_path)      # embedded, stored on disk
    atexit.register(client.close)                      # clean shutdown, no noisy tracebacks
    return client


def reset_client() -> None:
    """Close and forget the cached client (used by tests)."""
    if get_client.cache_info().currsize:
        get_client().close()
    get_client.cache_clear()


def collection_exists() -> bool:
    return get_client().collection_exists(get_settings().qdrant_collection)


def ensure_collection(dimension: int, recreate: bool = False) -> None:
    name = get_settings().qdrant_collection
    client = get_client()
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            # COSINE compares the direction of vectors, which is what we want for text
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def upsert_chunks(chunks, vectors) -> int:
    """Store vector + payload (text + source). Point id is derived from
    (source, chunk_index) so re-running ingestion overwrites instead of duplicating."""
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}#{c.chunk_index}")),
            vector=vec,
            payload={"text": c.text, "source": c.source, "chunk_index": c.chunk_index},
        )
        for c, vec in zip(chunks, vectors)
    ]
    get_client().upsert(collection_name=get_settings().qdrant_collection, points=points)
    return len(points)


def search(query_vector: list[float], top_k: int):
    """Similarity search. Returns Qdrant scored points (highest cosine score first)."""
    res = get_client().query_points(
        collection_name=get_settings().qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return res.points


def count() -> int:
    return get_client().count(get_settings().qdrant_collection).count
