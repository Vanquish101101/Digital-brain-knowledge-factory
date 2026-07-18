from click.testing import CliRunner

from kf.cli import cli
from kf.config import load_settings
from kf.store.postgres import connect


def test_stats_reports_document_and_chunk_counts():
    runner = CliRunner()

    result = runner.invoke(cli, ["stats"])

    assert result.exit_code == 0
    assert "Документов в Postgres" in result.output
    assert "Чанков" in result.output


def test_ingest_reports_summary(tmp_path, monkeypatch):
    (tmp_path / "cli-note.md").write_text("Заметка для теста CLI ingest.", encoding="utf-8")
    monkeypatch.setenv("SYNTHESIS_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setattr(
        "kf.ingest.synthesize_note",
        lambda settings, text, source_path: f"Заметка про {source_path}",
    )
    monkeypatch.setattr(
        "kf.ingest.extract_entities_and_relationships",
        lambda settings, text, source_path: ([], []),
    )

    # Clean up any leftover data from previous runs before testing
    settings = load_settings()
    conn = connect(settings)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE '%cli-note%'")
    conn.commit()
    conn.close()

    runner = CliRunner()

    result = runner.invoke(cli, ["ingest", "--source", str(tmp_path)])

    assert result.exit_code == 0
    assert "проиндексировано: 2" in result.output
    assert "заметок синтезировано: 1" in result.output
    assert "записей в журнале знаний" in result.output

    # Clean up after test
    conn = connect(settings)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE '%cli-note%'")
    conn.commit()
    conn.close()


def test_ingest_passes_detect_deletions_false_for_custom_source(tmp_path, monkeypatch):
    from kf.ingest import IngestStats

    captured = {}

    def _fake_ingest_directory(source_dir, deps, detect_deletions=True):
        captured["detect_deletions"] = detect_deletions
        return IngestStats()

    monkeypatch.setattr("kf.cli.ingest_directory", _fake_ingest_directory)

    runner = CliRunner()
    runner.invoke(cli, ["ingest", "--source", str(tmp_path)])

    assert captured["detect_deletions"] is False


def test_ingest_passes_detect_deletions_true_for_default_source(monkeypatch):
    from kf.ingest import IngestStats

    captured = {}

    def _fake_ingest_directory(source_dir, deps, detect_deletions=True):
        captured["detect_deletions"] = detect_deletions
        return IngestStats()

    monkeypatch.setattr("kf.cli.ingest_directory", _fake_ingest_directory)
    monkeypatch.setattr("kf.cli._build_ingest_deps", lambda settings: None)

    runner = CliRunner()
    runner.invoke(cli, ["ingest"])

    assert captured["detect_deletions"] is True


def test_graph_command_reports_relationship(tmp_path, monkeypatch):
    from kf.api import KnowledgeSession
    from kf.store.graph_store import add_relationship, ensure_schema, get_connection, upsert_entity

    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)
    upsert_entity(graph_conn, "Blender", "инструмент")
    upsert_entity(graph_conn, "Проект X", "проект")
    add_relationship(graph_conn, "Blender", "Проект X", "использует", "рендеринг сцен", "note.md")

    fake_session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )
    monkeypatch.setattr("kf.cli.open_session", lambda: fake_session)

    runner = CliRunner()
    result = runner.invoke(cli, ["graph", "Blender"])

    assert result.exit_code == 0
    assert "Проект X" in result.output
    assert "использует" in result.output


def test_graph_command_reports_not_found_for_unknown_entity(tmp_path, monkeypatch):
    from kf.api import KnowledgeSession
    from kf.store.graph_store import ensure_schema, get_connection

    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)

    fake_session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )
    monkeypatch.setattr("kf.cli.open_session", lambda: fake_session)

    runner = CliRunner()
    result = runner.invoke(cli, ["graph", "Несуществующая сущность"])

    assert result.exit_code == 0
    assert "не найдена" in result.output


def test_embedding_model_list_marks_active_profile(monkeypatch):
    monkeypatch.setattr("kf.cli.get_active_profile_name", lambda data_root: "local")
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: set())
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: set())
    monkeypatch.setattr("kf.cli.connect", lambda settings: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda settings: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "list"])

    assert result.exit_code == 0
    assert "* local" in result.output


def test_embedding_model_use_rejects_unknown_profile():
    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "use", "не-существует"])

    assert result.exit_code != 0
    assert "неизвестный профиль" in result.output


def test_embedding_model_use_switches_and_warns_about_gap(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: {"a.md", "b.md"})
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: {"a.md"})
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "use", "qwen3-8b"])

    assert result.exit_code == 0
    assert "Активная модель: qwen3-8b" in result.output
    assert "не хватает 1" in result.output


def test_embedding_model_sync_reports_up_to_date_when_nothing_missing(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.get_active_profile_name", lambda data_root: "local")
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: {"a.md"})
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: {"a.md"})
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_collection", lambda client, collection, vector_size: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "sync"])

    assert result.exit_code == 0
    assert "актуальна" in result.output


def test_embedding_model_sync_calls_sync_missing_paths(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.get_active_profile_name", lambda data_root: "local")
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: {"a.md", "b.md"})
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: {"a.md"})
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_collection", lambda client, collection, vector_size: None)
    monkeypatch.setattr("kf.cli.get_embedder_for_profile", lambda settings, profile: None)
    seen = {}

    def _fake_sync(missing_paths, source_dir, notes_dir, settings, profile, embedder, qdrant_client):
        seen["missing"] = missing_paths
        return (1, 0)

    monkeypatch.setattr("kf.cli.sync_missing_paths", _fake_sync)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "sync"])

    assert result.exit_code == 0
    assert seen["missing"] == {"b.md"}
    assert "синхронизировано: 1" in result.output
