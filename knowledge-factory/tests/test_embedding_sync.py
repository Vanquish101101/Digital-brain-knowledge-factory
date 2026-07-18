from pathlib import Path

import pytest

from kf.config import load_settings
from kf.embedding_models import EMBEDDING_PROFILES
from kf.embedding_sync import resolve_missing_path_text, sync_missing_paths
from kf.embeddings import embed_for_profile
from kf.store.qdrant_store import ensure_collection, get_client, list_paths

COLLECTION = "kf_test_sync"


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "source"
    d.mkdir()
    return d


@pytest.fixture
def notes_dir(tmp_path):
    d = tmp_path / "Синтезированные данные (synthesized-notes)"
    d.mkdir()
    return d


def test_resolve_missing_path_text_reads_source_file(source_dir, notes_dir):
    settings = load_settings()
    (source_dir / "заметка.md").write_text("Текст заметки", encoding="utf-8")

    text = resolve_missing_path_text("заметка.md", source_dir, notes_dir, settings)

    assert text == "Текст заметки"


def test_resolve_missing_path_text_reads_note_file(source_dir, notes_dir):
    settings = load_settings()
    (notes_dir / "заметка.md.md").write_text("Синтезированный текст", encoding="utf-8")
    rel_key = f"{notes_dir.name}/заметка.md.md"

    text = resolve_missing_path_text(rel_key, source_dir, notes_dir, settings)

    assert text == "Синтезированный текст"


def test_resolve_missing_path_text_returns_none_for_missing_file(source_dir, notes_dir):
    settings = load_settings()

    text = resolve_missing_path_text("не-существует.md", source_dir, notes_dir, settings)

    assert text is None


def test_sync_missing_paths_embeds_and_upserts(source_dir, notes_dir):
    settings = load_settings()
    profile = EMBEDDING_PROFILES["local"]
    (source_dir / "заметка.md").write_text("Текст про Blender и рендеринг сцен", encoding="utf-8")

    client = get_client(settings)
    ensure_collection(client, COLLECTION, vector_size=profile.dimension)
    try:
        synced, failed = sync_missing_paths(
            {"заметка.md"}, source_dir, notes_dir, settings, profile,
            embedder=None, qdrant_client=client, collection=COLLECTION,
        )

        assert synced == 1
        assert failed == 0
        assert list_paths(client, COLLECTION) == {"заметка.md"}
    finally:
        client.delete_collection(COLLECTION)


def test_sync_missing_paths_counts_failures_without_aborting(source_dir, notes_dir):
    settings = load_settings()
    profile = EMBEDDING_PROFILES["local"]
    (source_dir / "существует.md").write_text("Реальный текст файла", encoding="utf-8")

    client = get_client(settings)
    ensure_collection(client, COLLECTION, vector_size=profile.dimension)
    try:
        synced, failed = sync_missing_paths(
            {"существует.md", "не-существует.md"},
            source_dir, notes_dir, settings, profile,
            embedder=None, qdrant_client=client, collection=COLLECTION,
        )

        assert synced == 1
        assert failed == 1
    finally:
        client.delete_collection(COLLECTION)


def test_sync_missing_paths_isolates_single_file_embedding_failure(source_dir, notes_dir, monkeypatch):
    settings = load_settings()
    profile = EMBEDDING_PROFILES["local"]
    (source_dir / "рабочий.md").write_text("Текст рабочего файла про рендеринг сцен", encoding="utf-8")
    (source_dir / "сбойный.md").write_text("MARKER_FAIL текст файла с сетевым сбоем", encoding="utf-8")

    client = get_client(settings)
    ensure_collection(client, COLLECTION, vector_size=profile.dimension)
    try:
        real_embed_for_profile = embed_for_profile

        def fake_embed_for_profile(settings, profile, embedder, texts):
            if any("MARKER_FAIL" in text for text in texts):
                raise RuntimeError("симулированный сбой сети при эмбеддинге")
            return real_embed_for_profile(settings, profile, embedder, texts)

        monkeypatch.setattr("kf.embedding_sync.embed_for_profile", fake_embed_for_profile)

        synced, failed = sync_missing_paths(
            {"рабочий.md", "сбойный.md"},
            source_dir, notes_dir, settings, profile,
            embedder=None, qdrant_client=client, collection=COLLECTION,
        )

        assert synced == 1
        assert failed == 1
        assert list_paths(client, COLLECTION) == {"рабочий.md"}
    finally:
        client.delete_collection(COLLECTION)
