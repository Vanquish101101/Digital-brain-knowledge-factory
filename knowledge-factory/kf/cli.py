from datetime import date
from pathlib import Path

import click

from kf.api import ask_question, get_stats, graph_search, open_session, semantic_search
from kf.config import load_settings
from kf.deletion_sync import compute_deletion_candidates, purge_source
from kf.embedding_models import EMBEDDING_PROFILES, get_profile
from kf.embedding_state import get_active_profile_name, set_active_profile_name
from kf.embedding_sync import sync_missing_paths
from kf.embeddings import get_embedder_for_profile
from kf.ingest import IngestDeps, ingest_directory
from kf.journal import append_entries, detect_deleted, format_entry
from kf.scope import should_index
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.minio_store import ensure_bucket
from kf.store.minio_store import get_client as get_minio_client
from kf.store.postgres import connect, ensure_schema, list_paths, list_paths_with_hashes
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import list_paths as qdrant_list_paths
from kf.web_extract import derive_filename, extract_from_url

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "raw-data-repository"


def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    graph_conn = get_graph_connection(settings)
    ensure_graph_schema(graph_conn)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=session.profile.collection,
        settings=settings,
        graph_conn=graph_conn,
        profile=session.profile,
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


@cli.command(name="ingest-url")
@click.argument("url")
@click.option("--dest", required=True, help="Раздел vault (относительно raw-data-repository), куда сохранить.")
def ingest_url(url: str, dest: str):
    """Скачать статью или видео по ссылке, сохранить в vault и сразу проиндексировать."""
    settings = load_settings()
    try:
        text, title, is_low_quality = extract_from_url(url, settings)
    except Exception as exc:
        raise click.ClickException(f"Не удалось получить содержимое по ссылке: {exc}")

    dest_dir = DEFAULT_SOURCE / dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = derive_filename(title) + ".md"
    file_path = dest_dir / filename
    file_path.write_text(text, encoding="utf-8")

    click.echo(f"Сохранено: {dest}/{filename}")
    if is_low_quality:
        click.echo(
            "⚠ Извлечённый текст выглядит скудным — возможно, сайту нужен инструмент "
            "посильнее (Firecrawl, вручную)."
        )

    deps = _build_ingest_deps(settings)
    stats = ingest_directory(DEFAULT_SOURCE, deps, detect_deletions=False)
    click.echo(f"Готово. проиндексировано: {stats.files_ingested}, чанков записано: {stats.chunks_written}")


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


@cli.group(name="embedding-model")
def embedding_model():
    """Управление активной моделью эмбеддинга."""


@embedding_model.command(name="list")
def embedding_model_list():
    """Показать все профили моделей и их покрытие относительно текущей базы."""
    settings = load_settings()
    active = get_active_profile_name(settings.data_root)

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    known_paths = list_paths(pg_conn)

    qdrant_client = get_qdrant_client(settings)

    for name, profile in EMBEDDING_PROFILES.items():
        indexed_paths = qdrant_list_paths(qdrant_client, profile.collection)
        missing = detect_deleted(known_paths, indexed_paths)
        marker = "* " if name == active else "  "
        click.echo(
            f"{marker}{name} ({profile.provider}, {profile.model_id}, размерность={profile.dimension}): "
            f"{len(indexed_paths)}/{len(known_paths)} файлов, не хватает {len(missing)}"
        )


@embedding_model.command(name="use")
@click.argument("name")
def embedding_model_use(name: str):
    """Переключить активную модель эмбеддинга (без переиндексации)."""
    settings = load_settings()
    try:
        profile = get_profile(name)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    set_active_profile_name(settings.data_root, name)

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    known_paths = list_paths(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    indexed_paths = qdrant_list_paths(qdrant_client, profile.collection)
    missing = detect_deleted(known_paths, indexed_paths)

    click.echo(f"Активная модель: {name}")
    if missing:
        click.echo(
            f"⚠ Коллекция отстаёт: не хватает {len(missing)} файлов. "
            f"Запустите 'kf.py embedding-model sync', чтобы досчитать."
        )


@embedding_model.command(name="sync")
def embedding_model_sync():
    """Досчитать эмбеддинги только для файлов, которых не хватает в активной коллекции."""
    settings = load_settings()
    profile = get_profile(get_active_profile_name(settings.data_root))

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    known_paths = list_paths(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    ensure_collection(qdrant_client, profile.collection, vector_size=profile.dimension)
    indexed_paths = qdrant_list_paths(qdrant_client, profile.collection)
    missing = detect_deleted(known_paths, indexed_paths)

    if not missing:
        click.echo("Коллекция уже актуальна, синхронизация не нужна.")
        return

    embedder = get_embedder_for_profile(settings, profile)
    notes_dir = Path(settings.synthesis_notes_dir)
    synced, failed = sync_missing_paths(
        missing, DEFAULT_SOURCE, notes_dir, settings, profile, embedder, qdrant_client
    )
    click.echo(f"Готово. синхронизировано: {synced}, ошибок: {failed}")


@cli.command(name="sync-deletions")
@click.option("--yes", is_flag=True, default=False, help="Реально выполнить очистку (без флага — только план).")
def sync_deletions(yes: bool):
    """Найти файлы, реально удалённые из vault, и (по подтверждению) очистить осиротевшие записи."""
    settings = load_settings()
    pg_conn = connect(settings)
    ensure_schema(pg_conn)

    notes_dir_name = Path(settings.synthesis_notes_dir).name
    notes_prefix = f"{notes_dir_name}/"

    all_hashes = list_paths_with_hashes(pg_conn)
    known_source_hashes = list_paths_with_hashes(pg_conn, exclude_prefix=notes_prefix)

    seen_source_paths = {
        path.relative_to(DEFAULT_SOURCE).as_posix()
        for path in sorted(DEFAULT_SOURCE.rglob("*"))
        if path.is_file() and should_index(path)
    }

    confirmed, renamed = compute_deletion_candidates(known_source_hashes, seen_source_paths, all_hashes)

    if renamed:
        click.echo(f"Похоже на переименование/дубликат (не трогаем): {len(renamed)}")
        for p in renamed:
            click.echo(f"  - {p}")

    if not confirmed:
        click.echo("Нечего чистить — осиротевших записей не найдено.")
        return

    click.echo(f"К очистке: {len(confirmed)} файл(ов)")
    for p in confirmed:
        click.echo(f"  - {p}")

    if not yes:
        click.echo("\nЭто был предварительный просмотр. Запустите с --yes, чтобы реально очистить.")
        return

    qdrant_client = get_qdrant_client(settings)
    minio_client = get_minio_client(settings)
    graph_conn = get_graph_connection(settings)
    ensure_graph_schema(graph_conn)

    journal_entries = []
    today = date.today().isoformat()
    for p in confirmed:
        purge_source(p, settings, pg_conn, qdrant_client, minio_client, graph_conn)
        section = p.split("/", 1)[0]
        journal_entries.append(format_entry("очищено_из_базы", p, section, "", today))

    append_entries(journal_entries, DEFAULT_SOURCE.parent / "Журнал знаний.md")
    click.echo(f"Готово. Очищено: {len(confirmed)}")


if __name__ == "__main__":
    cli()
