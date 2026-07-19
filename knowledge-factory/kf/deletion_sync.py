from pathlib import Path

from kf.config import Settings
from kf.embedding_models import EMBEDDING_PROFILES
from kf.store.graph_store import delete_relationships_by_source
from kf.store.minio_store import remove_object
from kf.store.qdrant_store import delete_by_path


def compute_deletion_candidates(
    known_source_hashes: dict[str, str],
    seen_source_paths: set[str],
    all_known_hashes: dict[str, str],
) -> tuple[list[str], list[str]]:
    candidates = sorted(set(known_source_hashes) - seen_source_paths)
    seen_hashes = {h for p, h in all_known_hashes.items() if p in seen_source_paths}

    confirmed_deleted = []
    likely_renamed = []
    for path in candidates:
        if known_source_hashes[path] in seen_hashes:
            likely_renamed.append(path)
        else:
            confirmed_deleted.append(path)
    return confirmed_deleted, likely_renamed


def note_rel_key_for(source_rel_key: str, notes_dir_name: str) -> str:
    return f"{notes_dir_name}/{source_rel_key}.md"


def purge_source(
    source_rel_key: str,
    settings: Settings,
    pg_conn,
    qdrant_client,
    minio_client,
    graph_conn,
) -> None:
    notes_dir_name = Path(settings.synthesis_notes_dir).name
    note_rel_key = note_rel_key_for(source_rel_key, notes_dir_name)

    try:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE path IN (%s, %s)", (source_rel_key, note_rel_key))
        pg_conn.commit()
    except Exception as exc:
        print(f"[sync-deletions] Postgres не удалось очистить {source_rel_key}: {exc}")

    for profile in EMBEDDING_PROFILES.values():
        try:
            delete_by_path(qdrant_client, profile.collection, source_rel_key)
            delete_by_path(qdrant_client, profile.collection, note_rel_key)
        except Exception as exc:
            print(f"[sync-deletions] Qdrant ({profile.collection}) не удалось очистить {source_rel_key}: {exc}")

    for object_name in (source_rel_key, note_rel_key):
        try:
            remove_object(minio_client, object_name)
        except Exception as exc:
            print(f"[sync-deletions] MinIO не удалось очистить {object_name}: {exc}")

    try:
        delete_relationships_by_source(graph_conn, source_rel_key)
    except Exception as exc:
        print(f"[sync-deletions] граф не удалось очистить для {source_rel_key}: {exc}")

    note_path = Path(settings.synthesis_notes_dir) / f"{source_rel_key}.md"
    try:
        note_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[sync-deletions] не удалось удалить файл заметки {note_path}: {exc}")
