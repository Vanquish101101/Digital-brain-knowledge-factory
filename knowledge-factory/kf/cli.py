from pathlib import Path

import click

from kf.api import COLLECTION, VECTOR_SIZE, ask_question, get_stats, graph_search, open_session, semantic_search
from kf.config import load_settings
from kf.ingest import IngestDeps, ingest_directory
from kf.store.minio_store import ensure_bucket
from kf.store.minio_store import get_client as get_minio_client

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "raw-data-repository"


def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=COLLECTION,
        settings=settings,
        graph_conn=session.graph_conn,
    )


@click.group()
def cli():
    """Knowledge Factory CLI."""


@cli.command()
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка для индексации (по умолчанию — raw-data-repository).",
)
def ingest(source: Path | None):
    """Проиндексировать файлы: векторы -> Qdrant, текст -> Postgres, сырьё -> MinIO."""
    settings = load_settings()
    deps = _build_ingest_deps(settings)
    src = source or DEFAULT_SOURCE
    click.echo(f"Индексирую: {src}")
    stats = ingest_directory(src, deps, detect_deletions=(source is None))
    click.echo(
        f"Готово. просканировано: {stats.files_scanned}, "
        f"проиндексировано: {stats.files_ingested}, "
        f"пропущено (без изменений): {stats.files_skipped}, "
        f"ошибок: {stats.files_failed}, "
        f"чанков записано: {stats.chunks_written}, "
        f"заметок синтезировано: {stats.notes_synthesized}, "
        f"ошибок синтеза: {stats.notes_failed}, "
        f"записей в журнале знаний: {stats.journal_entries_written}"
    )
    if stats.deleted_detected:
        click.echo(
            f"⚠ Обнаружены удалённые файлы: {stats.deleted_detected}. "
            f"Проверьте 'Журнал знаний.md' — решение об очистке базы принимается отдельно."
        )


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Сколько фрагментов вернуть.")
def search(query: str, limit: int):
    """Найти релевантные фрагменты по смыслу запроса."""
    session = open_session()
    results = semantic_search(session, query, limit=limit)
    if not results:
        click.echo("Ничего не найдено.")
        return
    for r in results:
        click.echo(f"[{r['score']:.3f}] {r['path']} (чанк {r['chunk_index']})")
        click.echo(f"  {r['text'][:200]}")


@cli.command()
@click.argument("entity")
def graph(entity: str):
    """Показать все прямые связи сущности в графе знаний."""
    session = open_session()
    results = graph_search(session, entity)
    if not results:
        click.echo(f"Сущность «{entity}» не найдена в графе знаний.")
        return
    for r in results:
        click.echo(f"[{r['category']}] {r['entity']} — {r['description']} (из: {r['source_path']})")


@cli.command()
@click.argument("question")
@click.option("--limit", default=5, help="Сколько фрагментов подмешать в контекст.")
def ask(question: str, limit: int):
    """Задать вопрос базе знаний и получить связный ответ со ссылками на источники."""
    session = open_session()
    result = ask_question(session, question, limit=limit)
    click.echo(result["answer"])
    if result["sources"]:
        click.echo("\nИсточники:")
        for src in result["sources"]:
            click.echo(f"  - {src}")


@cli.command()
def stats():
    """Сколько документов и чанков сейчас в базе."""
    session = open_session()
    s = get_stats(session)
    click.echo(f"Документов в Postgres: {s['documents']}")
    click.echo(f"Чанков (векторов) в Qdrant: {s['chunks']}")


if __name__ == "__main__":
    cli()
