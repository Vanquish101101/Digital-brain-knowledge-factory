import uuid
from dataclasses import dataclass
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embeddings import embed
from kf.extract import extract_text
from kf.hashing import sha256_of_file
from kf.scope import should_index
from kf.store.minio_store import upload_file
from kf.store.postgres import needs_ingest, record_ingested
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


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_written: int = 0
    notes_synthesized: int = 0
    notes_failed: int = 0


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
) -> None:
    try:
        note_text = synthesize_note(deps.settings, text, rel_key)
    except Exception as exc:
        print(f"[ingest] синтез не удался для {rel_key}: {exc}")
        stats.notes_failed += 1
        return

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


def ingest_directory(source_dir: Path, deps: IngestDeps) -> IngestStats:
    stats = IngestStats()
    notes_dir = Path(deps.settings.synthesis_notes_dir).resolve()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or not should_index(path):
            continue

        rel_key = path.relative_to(source_dir).as_posix()
        stats.files_scanned += 1

        file_hash = sha256_of_file(path)
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
        if not is_note:
            _synthesize_and_index_note(text, rel_key, deps, stats)

    return stats
