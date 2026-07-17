import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embeddings import embed
from kf.extract import extract_text
from kf.graph import extract_entities_and_relationships
from kf.hashing import sha256_of_file
from kf.journal import append_entries, detect_deleted, extract_description, format_entry
from kf.scope import should_index
from kf.store.graph_store import add_relationship, upsert_entity
from kf.store.minio_store import upload_file
from kf.store.postgres import list_paths, needs_ingest, path_known, record_ingested
from kf.store.qdrant_store import upsert_chunks
from kf.synthesize import synthesize_note

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


@dataclass
class IngestDeps:
    pg_conn: object
    qdrant_client: object
    minio_client: object
    embedder: object
    collection: str
    settings: Settings
    max_chars: int = 1500
    overlap: int = 150
    graph_conn: object = None


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_written: int = 0
    notes_synthesized: int = 0
    notes_failed: int = 0
    journal_entries_written: int = 0
    deleted_detected: int = 0
    entities_extracted: int = 0
    entities_failed: int = 0


def _point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}:{chunk_index}"))


def _store_text(text: str, rel_key: str, path: Path, deps: IngestDeps, stats: IngestStats) -> None:
    chunks = chunk_text(text, max_chars=deps.max_chars, overlap=deps.overlap)
    if chunks:
        vectors = embed(deps.embedder, chunks)
        points = [
            {
                "id": _point_id(rel_key, i),
                "vector": vectors[i],
                "payload": {"path": rel_key, "chunk_index": i, "text": chunks[i]},
            }
            for i in range(len(chunks))
        ]
        upsert_chunks(deps.qdrant_client, deps.collection, points)
        stats.chunks_written += len(points)
    upload_file(deps.minio_client, path, rel_key)


def _synthesize_and_index_note(
    text: str, rel_key: str, deps: IngestDeps, stats: IngestStats
) -> str | None:
    try:
        note_text = synthesize_note(deps.settings, text, rel_key)
    except Exception as exc:
        print(f"[ingest] синтез не удался для {rel_key}: {exc}")
        stats.notes_failed += 1
        return None

    notes_dir = Path(deps.settings.synthesis_notes_dir)
    note_path = notes_dir / f"{rel_key}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    stats.notes_synthesized += 1

    note_rel_key = f"{notes_dir.name}/{rel_key}.md"
    note_hash = sha256_of_file(note_path)
    if needs_ingest(deps.pg_conn, note_rel_key, note_hash):
        _store_text(note_text, note_rel_key, note_path, deps, stats)
        record_ingested(deps.pg_conn, note_rel_key, note_hash)
        stats.files_ingested += 1

    return note_text


def _extract_and_store_entities(text: str, rel_key: str, deps: IngestDeps, stats: IngestStats) -> None:
    if deps.graph_conn is None:
        return
    try:
        entities, relationships = extract_entities_and_relationships(deps.settings, text, rel_key)
    except Exception as exc:
        print(f"[ingest] извлечение сущностей не удалось для {rel_key}: {exc}")
        stats.entities_failed += 1
        return

    for entity in entities:
        upsert_entity(deps.graph_conn, entity["name"], entity["type"])
    for rel in relationships:
        add_relationship(
            deps.graph_conn, rel["from"], rel["to"], rel["category"], rel["description"], rel_key
        )
    stats.entities_extracted += len(entities)


def ingest_directory(source_dir: Path, deps: IngestDeps, detect_deletions: bool = True) -> IngestStats:
    stats = IngestStats()
    notes_dir = Path(deps.settings.synthesis_notes_dir).resolve()
    journal_entries: list[str] = []
    seen_paths: set[str] = set()
    today = date.today().isoformat()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or not should_index(path):
            continue

        rel_key = path.relative_to(source_dir).as_posix()
        stats.files_scanned += 1
        seen_paths.add(rel_key)

        file_hash = sha256_of_file(path)
        is_new = not path_known(deps.pg_conn, rel_key)
        if not needs_ingest(deps.pg_conn, rel_key, file_hash):
            stats.files_skipped += 1
            continue

        try:
            text = extract_text(path, deps.settings)
        except Exception as exc:
            print(f"[ingest] пропускаю {rel_key}: {exc}")
            stats.files_failed += 1
            continue

        _store_text(text, rel_key, path, deps, stats)
        record_ingested(deps.pg_conn, rel_key, file_hash)
        stats.files_ingested += 1

        is_note = notes_dir in path.resolve().parents
        note_text = None
        if not is_note:
            note_text = _synthesize_and_index_note(text, rel_key, deps, stats)
            _extract_and_store_entities(text, rel_key, deps, stats)

        section = rel_key.split("/", 1)[0]
        description = extract_description(note_text) if note_text else ""
        action = "добавлено" if is_new else "изменено"
        journal_entries.append(format_entry(action, rel_key, section, description, today))

    if detect_deletions:
        notes_prefix = f"{notes_dir.name}/"
        try:
            known_paths = list_paths(deps.pg_conn, exclude_prefix=notes_prefix)
            deleted = detect_deleted(known_paths, seen_paths)
            for deleted_path in sorted(deleted):
                section = deleted_path.split("/", 1)[0]
                journal_entries.append(format_entry("удалено", deleted_path, section, "", today))
            stats.deleted_detected = len(deleted)
        except Exception as exc:
            print(f"[ingest] проверка удалённых файлов не удалась: {exc}")

    stats.journal_entries_written = len(journal_entries)
    try:
        append_entries(journal_entries, source_dir.parent / "Журнал знаний.md")
    except Exception as exc:
        print(f"[ingest] запись в Журнал знаний.md не удалась: {exc}")

    return stats
