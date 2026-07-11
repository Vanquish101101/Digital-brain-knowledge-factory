from pathlib import Path

import click

from kf.api import COLLECTION, VECTOR_SIZE, ask_question, get_stats, open_session, semantic_search
from kf.config import load_settings
from kf.ingest import IngestDeps, ingest_directory
from kf.store.minio_store import ensure_bucket
from kf.store.minio_store import get_client as get_minio_client

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "Хранилище входных данных"


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
    )


@click.group()
def cli():
    """Knowledge Factory CLI."""


@cli.command()
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка для индексации (по умолчанию — Хранилище входных данных).",
)
def ingest(source: Path | None):
    """Проиндексировать файлы: векторы -> Qdrant, текст -> Postgres, сырьё -> MinIO."""
    settings = load_settings()
    deps = _build_ingest_deps(settings)
    src = source or DEFAULT_SOURCE
    click.echo(f"Индексирую: {src}")
    stats = ingest_directory(src, deps)
    click.echo(
        f"Готово. просканировано: {stats.files_scanned}, "
        f"проиндексировано: {stats.files_ingested}, "
        f"пропущено (без изменений): {stats.files_skipped}, "
        f"чанков записано: {stats.chunks_written}"
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
