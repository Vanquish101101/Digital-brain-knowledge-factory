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
