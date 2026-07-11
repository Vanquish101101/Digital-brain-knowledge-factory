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


def test_ingest_reports_summary(tmp_path):
    (tmp_path / "cli-note.md").write_text("Заметка для теста CLI ingest.", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli, ["ingest", "--source", str(tmp_path)])

    assert result.exit_code == 0
    assert "проиндексировано: 1" in result.output

    settings = load_settings()
    conn = connect(settings)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path = 'cli-note.md'")
    conn.commit()
    conn.close()
