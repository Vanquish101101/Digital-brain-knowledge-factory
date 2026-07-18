import uuid

from kf.api import ask_question, get_stats, open_session, semantic_search
from kf.embedding_models import EMBEDDING_PROFILES
from kf.store.postgres import needs_ingest, record_ingested
from kf.store.qdrant_store import delete_by_path, upsert_chunks
from kf.embeddings import embed


def _seed_one_chunk(session, path, text):
    vector = embed(session.embedder, [text])[0]
    upsert_chunks(
        session.qdrant_client,
        "knowledge",
        [{"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": path, "chunk_index": 0, "text": text}}],
    )


def test_semantic_search_finds_seeded_chunk():
    session = open_session()
    path = f"test://api-{uuid.uuid4()}.md"
    _seed_one_chunk(session, path, "Рецепт борща с говядиной и свёклой.")

    try:
        results = semantic_search(session, "как приготовить борщ", limit=3)
        assert any(r["path"] == path for r in results)
    finally:
        delete_by_path(session.qdrant_client, "knowledge", path)


def test_ask_question_returns_answer_and_sources():
    session = open_session()
    path = f"test://api-{uuid.uuid4()}.md"
    _seed_one_chunk(session, path, "Столица Франции — Париж.")

    try:
        result = ask_question(session, "Какая столица у Франции?", limit=3)
        assert "answer" in result
        assert isinstance(result["answer"], str) and len(result["answer"]) > 0
        assert path in result["sources"]
    finally:
        delete_by_path(session.qdrant_client, "knowledge", path)


def test_get_stats_returns_document_and_chunk_counts():
    session = open_session()

    stats = get_stats(session)

    assert "documents" in stats
    assert "chunks" in stats
    assert stats["documents"] >= 0
    assert stats["chunks"] >= 0


from kf.api import KnowledgeSession, graph_search
from kf.config import load_settings
from kf.store.graph_store import add_relationship, ensure_schema, get_connection, upsert_entity


def test_graph_search_finds_direct_relationship(tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)
    upsert_entity(graph_conn, "Blender", "инструмент")
    upsert_entity(graph_conn, "Проект X", "проект")
    add_relationship(graph_conn, "Blender", "Проект X", "использует", "тестовая связь", "test.md")

    session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )

    results = graph_search(session, "Blender")

    assert len(results) == 1
    assert results[0]["entity"] == "Проект X"


def test_graph_search_returns_empty_for_unknown_entity(tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)

    session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )

    results = graph_search(session, "Несуществующая сущность")

    assert results == []


def test_open_session_does_not_eagerly_open_graph_connection():
    session = open_session()

    assert session.graph_conn is None


def test_graph_search_opens_and_releases_its_own_connection_when_session_has_none(tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=None
    )

    results = graph_search(session, "Несуществующая сущность")

    assert results == []
    # A second, independent connection to the same path must succeed — proves the first
    # connection graph_search opened internally was actually released, not leaked.
    from kf.store.graph_store import ensure_schema, get_connection

    second_conn = get_connection(settings)
    ensure_schema(second_conn)


def test_open_session_defaults_to_local_profile():
    session = open_session()

    assert session.profile.name == "local"
    assert session.profile.collection == "knowledge"


def test_semantic_search_uses_session_profile_collection(monkeypatch):
    session = open_session()
    calls = []
    monkeypatch.setattr(
        "kf.api.qdrant_search",
        lambda client, collection, vector, limit: calls.append(collection) or [],
    )

    semantic_search(session, "тест", limit=1)

    assert calls == [session.profile.collection]
