import pytest

from kf.config import load_settings
from kf.embeddings import get_embedder
from kf.ingest import IngestDeps, ingest_directory
from kf.store.minio_store import ensure_bucket, file_exists, get_client as get_minio_client
from kf.store.postgres import connect, ensure_schema
from kf.store.qdrant_store import ensure_collection, get_client as get_qdrant_client, search

COLLECTION = "kf_test_ingest"


@pytest.fixture
def deps():
    settings = load_settings()

    pg_conn = connect(settings)
    ensure_schema(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    ensure_collection(qdrant_client, COLLECTION, vector_size=384)

    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)

    embedder = get_embedder(settings)

    d = IngestDeps(
        pg_conn=pg_conn,
        qdrant_client=qdrant_client,
        minio_client=minio_client,
        embedder=embedder,
        collection=COLLECTION,
    )
    yield d

    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE '%note1%.md' OR path LIKE 'note%.md'")
    pg_conn.commit()
    qdrant_client.delete_collection(COLLECTION)
    pg_conn.close()


def test_ingests_new_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка про борщ и его рецепт.", encoding="utf-8")
    (tmp_path / "note2.md").write_text("Заметка про Docker и контейнеры.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 2
    assert stats.files_ingested == 2
    assert stats.files_skipped == 0
    assert stats.chunks_written == 2
    assert file_exists(deps.minio_client, "note1.md") is True
    assert file_exists(deps.minio_client, "note2.md") is True


def test_second_run_skips_unchanged_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка про борщ и его рецепт.", encoding="utf-8")

    ingest_directory(tmp_path, deps)
    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 0
    assert stats.files_skipped == 1


def test_changed_file_gets_reingested(tmp_path, deps):
    f = tmp_path / "note1.md"
    f.write_text("Старый текст.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.write_text("Новый текст после правки.", encoding="utf-8")
    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested == 1
    assert stats.files_skipped == 0


def test_ignores_excluded_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")
    (tmp_path / "video.mp4").write_bytes(b"not a real video")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 1


def test_nested_file_uses_forward_slash_object_key(tmp_path, deps):
    subdir = tmp_path / "001 Подпапка (со скобками)"
    subdir.mkdir()
    (subdir / "note1.md").write_text("Вложенная заметка.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    assert file_exists(deps.minio_client, "001 Подпапка (со скобками)/note1.md") is True
