import uuid

import pytest

from kf.config import load_settings
from kf.embeddings import embed, get_embedder
from kf.store.qdrant_store import delete_by_path, ensure_collection, get_client, search, upsert_chunks

COLLECTION = "kf_test"


@pytest.fixture
def client():
    settings = load_settings()
    c = get_client(settings)
    ensure_collection(c, COLLECTION, vector_size=384)
    yield c
    c.delete_collection(COLLECTION)


@pytest.fixture
def embedder():
    settings = load_settings()
    return get_embedder(settings)


def test_search_finds_semantically_similar_chunk(client, embedder):
    texts = ["Рецепт борща с говядиной", "Настройка Docker-контейнера для Postgres"]
    vectors = embed(embedder, texts)
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": vectors[0],
            "payload": {"path": "test://borscht.md", "chunk_index": 0, "text": texts[0]},
        },
        {
            "id": str(uuid.uuid4()),
            "vector": vectors[1],
            "payload": {"path": "test://docker.md", "chunk_index": 0, "text": texts[1]},
        },
    ]

    upsert_chunks(client, COLLECTION, points)

    query_vector = embed(embedder, ["как приготовить борщ"])[0]
    results = search(client, COLLECTION, query_vector, limit=1)

    assert len(results) == 1
    assert results[0]["payload"]["path"] == "test://borscht.md"


def test_upsert_same_id_overwrites(client, embedder):
    vector = embed(embedder, ["первая версия"])[0]
    point_id = str(uuid.uuid4())
    upsert_chunks(
        client,
        COLLECTION,
        [{"id": point_id, "vector": vector, "payload": {"path": "test://a.md", "chunk_index": 0, "text": "старое"}}],
    )
    upsert_chunks(
        client,
        COLLECTION,
        [{"id": point_id, "vector": vector, "payload": {"path": "test://a.md", "chunk_index": 0, "text": "новое"}}],
    )

    results = search(client, COLLECTION, vector, limit=10)
    matching = [r for r in results if r["payload"]["path"] == "test://a.md"]

    assert len(matching) == 1
    assert matching[0]["payload"]["text"] == "новое"


def test_upserts_many_points_in_batches(client, embedder):
    vector = embed(embedder, ["один и тот же вектор для скорости"])[0]
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {"path": f"test://bulk-{i}.md", "chunk_index": 0, "text": "x"},
        }
        for i in range(150)
    ]

    upsert_chunks(client, COLLECTION, points, batch_size=64)

    count = client.count(COLLECTION).count
    assert count == 150


def test_delete_by_path_removes_only_matching_points(client, embedder):
    vector = embed(embedder, ["текст"])[0]
    upsert_chunks(
        client,
        COLLECTION,
        [
            {"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": "test://keep.md", "chunk_index": 0, "text": "a"}},
            {"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": "test://remove.md", "chunk_index": 0, "text": "b"}},
            {"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": "test://remove.md", "chunk_index": 1, "text": "c"}},
        ],
    )

    delete_by_path(client, COLLECTION, "test://remove.md")

    remaining = [r["payload"]["path"] for r in search(client, COLLECTION, vector, limit=10)]
    assert remaining == ["test://keep.md"]
