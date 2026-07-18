from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embedding_models import EmbeddingProfile
from kf.embeddings import embed_for_profile, get_embedder_for_profile
from kf.extract import extract_text
from kf.store.qdrant_store import point_id, upsert_chunks

_MAX_CHARS = 1500
_OVERLAP = 150


def resolve_missing_path_text(
    rel_key: str, source_dir: Path, notes_dir: Path, settings: Settings
) -> str | None:
    notes_prefix = f"{notes_dir.name}/"
    if rel_key.startswith(notes_prefix):
        note_path = notes_dir.parent / rel_key
        if not note_path.exists():
            return None
        return note_path.read_text(encoding="utf-8")

    source_path = source_dir / rel_key
    if not source_path.exists():
        return None
    return extract_text(source_path, settings)


def sync_missing_paths(
    missing_paths: set[str],
    source_dir: Path,
    notes_dir: Path,
    settings: Settings,
    profile: EmbeddingProfile,
    embedder: object,
    qdrant_client: object,
    collection: str | None = None,
) -> tuple[int, int]:
    target_collection = collection or profile.collection
    resolved_embedder = embedder if embedder is not None else get_embedder_for_profile(settings, profile)
    synced = 0
    failed = 0

    for rel_key in sorted(missing_paths):
        try:
            text = resolve_missing_path_text(rel_key, source_dir, notes_dir, settings)
            if text is None:
                print(f"[embedding-sync] файл не найден на диске: {rel_key}")
                failed += 1
                continue

            chunks = chunk_text(text, max_chars=_MAX_CHARS, overlap=_OVERLAP)
            if not chunks:
                synced += 1
                continue

            vectors = embed_for_profile(settings, profile, resolved_embedder, chunks)
            points = [
                {
                    "id": point_id(rel_key, i),
                    "vector": vectors[i],
                    "payload": {"path": rel_key, "chunk_index": i, "text": chunks[i]},
                }
                for i in range(len(chunks))
            ]
            upsert_chunks(qdrant_client, target_collection, points)
            synced += 1
        except Exception as exc:
            print(f"[embedding-sync] не удалось синхронизировать {rel_key}: {exc}")
            failed += 1

    return synced, failed
