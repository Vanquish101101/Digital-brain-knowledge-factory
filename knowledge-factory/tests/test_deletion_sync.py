import uuid
from pathlib import Path

from kf.config import load_settings
from kf.deletion_sync import compute_deletion_candidates, note_rel_key_for, purge_source
from kf.embedding_models import EmbeddingProfile
from kf.store.graph_store import add_relationship, ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.graph_store import query_entity, upsert_entity
from kf.store.minio_store import ensure_bucket, file_exists
from kf.store.minio_store import get_client as get_minio_client
from kf.store.minio_store import upload_file
from kf.store.postgres import connect, ensure_schema, path_known, record_ingested
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import list_paths as qdrant_list_paths
from kf.store.qdrant_store import upsert_chunks


def test_compute_deletion_candidates_flags_missing_path_as_deleted():
    known = {"a.md": "hash-a", "b.md": "hash-b"}
    seen = {"a.md"}
    all_hashes = {"a.md": "hash-a", "b.md": "hash-b"}

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == ["b.md"]
    assert renamed == []


def test_compute_deletion_candidates_detects_rename_by_matching_hash():
    known = {"old.md": "hash-x"}
    seen = {"new.md"}
    all_hashes = {"old.md": "hash-x", "new.md": "hash-x"}

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == []
    assert renamed == ["old.md"]


def test_compute_deletion_candidates_returns_empty_when_nothing_missing():
    known = {"a.md": "hash-a"}
    seen = {"a.md"}
    all_hashes = {"a.md": "hash-a"}

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == []
    assert renamed == []


def test_compute_deletion_candidates_sorts_results():
    known = {"z.md": "h1", "a.md": "h2"}
    seen: set[str] = set()
    all_hashes = known

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == ["a.md", "z.md"]


def test_note_rel_key_for_builds_expected_path():
    assert (
        note_rel_key_for("003 Знания/файл.md", "synthesized-notes")
        == "synthesized-notes/003 Знания/файл.md.md"
    )


def test_purge_source_removes_from_all_stores(tmp_path, monkeypatch):
    settings = load_settings()
    settings.synthesis_notes_dir = str(tmp_path / "notes")
    settings.data_root = str(tmp_path)
    Path(settings.synthesis_notes_dir).mkdir(parents=True, exist_ok=True)

    source_rel_key = "test-purge/файл.md"
    notes_dir_name = Path(settings.synthesis_notes_dir).name
    note_rel_key = note_rel_key_for(source_rel_key, notes_dir_name)

    # Фейковые профили с выделенными тестовыми коллекциями — НЕ трогаем реальную
    # продакшен-коллекцию "knowledge". Два профиля — чтобы доказать, что purge_source
    # реально чистит каждую коллекцию по очереди, а не только одну.
    test_collections = ["kf_test_deletion_sync_a", "kf_test_deletion_sync_b"]
    fake_profiles = {
        name: EmbeddingProfile(name=name, provider="local", model_id="m", dimension=8, collection=collection)
        for name, collection in zip(("test-a", "test-b"), test_collections)
    }
    monkeypatch.setattr("kf.deletion_sync.EMBEDDING_PROFILES", fake_profiles)

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    record_ingested(pg_conn, source_rel_key, "hash-x")
    record_ingested(pg_conn, note_rel_key, "hash-y")

    qdrant_client = get_qdrant_client(settings)
    vector = [0.0] * 8
    try:
        for collection in test_collections:
            ensure_collection(qdrant_client, collection, vector_size=8)
            upsert_chunks(
                qdrant_client,
                collection,
                [{"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": source_rel_key, "chunk_index": 0, "text": "x"}}],
            )

        minio_client = get_minio_client(settings)
        ensure_bucket(minio_client)
        raw_file = tmp_path / "raw.md"
        raw_file.write_text("тест", encoding="utf-8")
        upload_file(minio_client, raw_file, source_rel_key)

        graph_conn = get_graph_connection(settings)
        ensure_graph_schema(graph_conn)
        upsert_entity(graph_conn, "ТестоваяСущность", "концепт")
        upsert_entity(graph_conn, "ДругаяСущность", "концепт")
        add_relationship(graph_conn, "ТестоваяСущность", "ДругаяСущность", "связано_с_темой", "тест", source_rel_key)

        note_path = Path(settings.synthesis_notes_dir) / f"{source_rel_key}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("заметка", encoding="utf-8")

        purge_source(source_rel_key, settings, pg_conn, qdrant_client, minio_client, graph_conn)

        assert path_known(pg_conn, source_rel_key) is False
        assert path_known(pg_conn, note_rel_key) is False
        for collection in test_collections:
            assert qdrant_list_paths(qdrant_client, collection) == set()
        assert file_exists(minio_client, source_rel_key) is False
        assert file_exists(minio_client, note_rel_key) is False
        assert query_entity(graph_conn, "ТестоваяСущность") == []
        assert not note_path.exists()
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE path IN (%s, %s)", (source_rel_key, note_rel_key))
        pg_conn.commit()
        for collection in test_collections:
            try:
                qdrant_client.delete_collection(collection)
            except Exception:
                pass
