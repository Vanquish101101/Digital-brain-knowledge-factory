from pathlib import Path

import pytest

from kf.config import load_settings
from kf.embeddings import get_embedder
from kf.ingest import IngestDeps, ingest_directory
from kf.store.minio_store import ensure_bucket, file_exists, get_client as get_minio_client
from kf.store.postgres import connect, ensure_schema
from kf.store.qdrant_store import ensure_collection, get_client as get_qdrant_client

COLLECTION = "kf_test_ingest"


@pytest.fixture
def deps(tmp_path_factory, monkeypatch):
    settings = load_settings()
    settings.synthesis_notes_dir = str(tmp_path_factory.mktemp("synth-notes"))
    monkeypatch.setattr(
        "kf.ingest.synthesize_note",
        lambda settings, text, source_path: f"Синтез-заметка про {source_path}",
    )

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
        settings=settings,
    )
    yield d

    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE '%note%'")
    pg_conn.commit()
    qdrant_client.delete_collection(COLLECTION)
    pg_conn.close()


def test_ingests_new_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка про борщ и его рецепт.", encoding="utf-8")
    (tmp_path / "note2.md").write_text("Заметка про Docker и контейнеры.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 2
    assert stats.files_ingested == 4
    assert stats.files_skipped == 0
    assert stats.chunks_written == 4
    assert stats.notes_synthesized == 2
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
    (tmp_path / "archive.zip").write_bytes(b"not a real archive")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 2
    assert stats.notes_synthesized == 1


def test_nested_file_uses_forward_slash_object_key(tmp_path, deps):
    subdir = tmp_path / "001 Подпапка (со скобками)"
    subdir.mkdir()
    (subdir / "note1.md").write_text("Вложенная заметка.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    assert file_exists(deps.minio_client, "001 Подпапка (со скобками)/note1.md") is True


def test_failed_file_does_not_abort_whole_run(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")
    (tmp_path / "broken.png").write_bytes(b"not a real png")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 2
    assert stats.files_ingested == 2
    assert stats.files_failed == 1


def test_writes_synthesized_note_file_to_notes_dir(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    note_path = Path(deps.settings.synthesis_notes_dir) / "note1.md.md"
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "Синтез-заметка про note1.md"


def test_note_object_key_follows_configured_notes_dir_name(tmp_path, deps):
    custom_notes_dir = tmp_path.parent / "custom-notes-dir"
    deps.settings.synthesis_notes_dir = str(custom_notes_dir)
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    assert file_exists(deps.minio_client, "custom-notes-dir/note1.md.md") is True


def test_synthesis_failure_does_not_lose_source_chunks(tmp_path, deps, monkeypatch):
    (tmp_path / "note1.md").write_text("Годная заметка, но синтез упадёт.", encoding="utf-8")

    def _boom(settings, text, source_path):
        raise RuntimeError("OpenRouter недоступен")

    monkeypatch.setattr("kf.ingest.synthesize_note", _boom)

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 1
    assert stats.chunks_written == 1
    assert stats.notes_failed == 1
    assert stats.notes_synthesized == 0


def test_files_inside_notes_dir_are_not_resynthesized(tmp_path, deps):
    notes_dir = Path(deps.settings.synthesis_notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "already-a-note.md").write_text("Уже готовая синтез-заметка.", encoding="utf-8")

    stats = ingest_directory(notes_dir, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 1
    assert stats.notes_synthesized == 0


def test_journal_logs_added_then_changed_actions(tmp_path, deps):
    f = tmp_path / "note-journal.md"
    f.write_text("Первая версия заметки.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.write_text("Изменённая версия заметки.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    journal_path = tmp_path.parent / "Журнал знаний.md"
    content = journal_path.read_text(encoding="utf-8")
    assert "добавлено | note-journal.md" in content
    assert "изменено | note-journal.md" in content


def test_deleted_file_is_logged_in_journal(tmp_path, deps):
    f = tmp_path / "note-to-delete.md"
    f.write_text("Заметка, которую скоро удалим.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.unlink()
    stats = ingest_directory(tmp_path, deps)

    journal_path = tmp_path.parent / "Журнал знаний.md"
    content = journal_path.read_text(encoding="utf-8")
    assert "удалено | note-to-delete.md" in content
    assert stats.deleted_detected >= 1


def test_journal_entry_uses_synthesis_note_first_line_as_description(tmp_path, deps, monkeypatch):
    monkeypatch.setattr(
        "kf.ingest.synthesize_note",
        lambda settings, text, source_path: "1. Заметка о рецепте борща.\n2. Ключевые идеи...",
    )
    (tmp_path / "note-desc.md").write_text("Заметка про борщ.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    journal_path = tmp_path.parent / "Журнал знаний.md"
    content = journal_path.read_text(encoding="utf-8")
    assert "Заметка о рецепте борща." in content


def test_deletion_check_failure_does_not_abort_ingest(tmp_path, deps, monkeypatch):
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    def _boom(conn, exclude_prefix=""):
        raise RuntimeError("Postgres недоступен")

    monkeypatch.setattr("kf.ingest.list_paths", _boom)

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested == 2


def test_journal_write_failure_does_not_abort_ingest(tmp_path, deps, monkeypatch):
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    def _boom(entries, path):
        raise OSError("файл занят другим процессом")

    monkeypatch.setattr("kf.ingest.append_entries", _boom)

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested == 2


def test_detect_deletions_false_skips_deletion_check(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps, detect_deletions=False)

    assert stats.deleted_detected == 0


def test_ingest_skips_entity_extraction_when_graph_conn_is_none(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.entities_extracted == 0
    assert stats.entities_failed == 0


def test_ingest_extracts_entities_when_graph_conn_provided(tmp_path, deps, monkeypatch):
    from kf.store.graph_store import ensure_schema, get_connection, query_entity

    graph_settings = deps.settings
    graph_settings.data_root = str(tmp_path.parent / "graph-data-extract")
    deps.graph_conn = get_connection(graph_settings)
    ensure_schema(deps.graph_conn)

    monkeypatch.setattr(
        "kf.ingest.extract_entities_and_relationships",
        lambda settings, text, source_path: (
            [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}],
            [{"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг"}],
        ),
    )
    (tmp_path / "note1.md").write_text("Заметка про Blender и Проект X.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.entities_extracted == 2
    results = query_entity(deps.graph_conn, "Blender")
    assert len(results) == 1
    assert results[0]["entity"] == "Проект X"


def test_reingest_replaces_relationships_instead_of_duplicating(tmp_path, deps, monkeypatch):
    from kf.store.graph_store import ensure_schema, get_connection, query_entity

    graph_settings = deps.settings
    graph_settings.data_root = str(tmp_path.parent / "graph-data-reingest")
    deps.graph_conn = get_connection(graph_settings)
    ensure_schema(deps.graph_conn)

    monkeypatch.setattr(
        "kf.ingest.extract_entities_and_relationships",
        lambda settings, text, source_path: (
            [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}],
            [{"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг"}],
        ),
    )
    f = tmp_path / "note1.md"
    f.write_text("Первая версия.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.write_text("Изменённая версия.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    results = query_entity(deps.graph_conn, "Blender")
    assert len(results) == 1


def test_entity_extraction_failure_does_not_abort_ingest(tmp_path, deps, monkeypatch):
    from kf.store.graph_store import ensure_schema, get_connection

    graph_settings = deps.settings
    graph_settings.data_root = str(tmp_path.parent / "graph-data-failure")
    deps.graph_conn = get_connection(graph_settings)
    ensure_schema(deps.graph_conn)

    def _boom(settings, text, source_path):
        raise RuntimeError("OpenRouter недоступен")

    monkeypatch.setattr("kf.ingest.extract_entities_and_relationships", _boom)
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested == 2
    assert stats.entities_failed == 1
