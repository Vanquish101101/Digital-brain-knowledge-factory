import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from kf.config import Settings


def get_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=60)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, collection: str, points: list[dict], batch_size: int = 64) -> None:
    structs = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points]
    for i in range(0, len(structs), batch_size):
        client.upsert(collection_name=collection, points=structs[i : i + batch_size])


def delete_by_path(client: QdrantClient, collection: str, path: str) -> None:
    client.delete(
        collection_name=collection,
        points_selector=Filter(must=[FieldCondition(key="path", match=MatchValue(value=path))]),
    )


def search(client: QdrantClient, collection: str, query_vector: list[float], limit: int = 5) -> list[dict]:
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
    ).points
    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]


_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}:{chunk_index}"))


def list_paths(client: QdrantClient, collection: str) -> set[str]:
    if not client.collection_exists(collection):
        return set()

    paths: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_payload=["path"],
            with_vectors=False,
            offset=offset,
        )
        paths.update(p.payload["path"] for p in points)
        if offset is None:
            break
    return paths
