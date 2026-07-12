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


def _point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}:{chunk_index}"))


def ingest_directory(source_dir: Path, deps: IngestDeps) -> IngestStats:
    stats = IngestStats()

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
        record_ingested(deps.pg_conn, rel_key, file_hash)
        stats.files_ingested += 1

    return stats
