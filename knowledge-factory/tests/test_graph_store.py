import pytest

from kf.config import Settings
from kf.store.graph_store import (
    add_relationship,
    delete_relationships_by_source,
    ensure_schema,
    get_connection,
    normalize_entity_name,
    query_entity,
    upsert_entity,
)


def _dummy_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        postgres_host="localhost", postgres_port=5432, postgres_user="u",
        postgres_password="p", postgres_db="d", qdrant_url="http://localhost:6333",
        minio_endpoint="localhost:9000", minio_access_key="a", minio_secret_key="s",
        data_root=str(tmp_path), model_cache_dir=str(tmp_path / "model-cache"),
        embedding_model="m", openrouter_api_key="k", llm_model="l", ocr_languages="eng",
        image_caption_threshold_chars=20, vision_model="v",
        video_frame_interval_seconds=15, whisper_model_size="small",
        max_video_frames=20, synthesis_notes_dir=str(tmp_path / "notes"),
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def conn(tmp_path):
    settings = _dummy_settings(tmp_path)
    connection = get_connection(settings)
    ensure_schema(connection)
    return connection


def test_normalize_entity_name_trims_and_lowercases():
    assert normalize_entity_name("  Blender  ") == "blender"
    assert normalize_entity_name("BLENDER") == "blender"


def test_query_entity_returns_empty_for_unknown_entity(conn):
    results = query_entity(conn, "Несуществующая сущность")

    assert results == []


def test_upsert_entity_is_idempotent_across_case_and_spacing(conn):
    upsert_entity(conn, "Blender", "инструмент")
    upsert_entity(conn, "blender", "инструмент")
    upsert_entity(conn, "  BLENDER  ", "инструмент")
    upsert_entity(conn, "Проект X", "проект")
    add_relationship(conn, "Blender", "Проект X", "использует", "рендеринг сцен", "заметка.md")

    results = query_entity(conn, "blender")

    assert len(results) == 1


def test_add_relationship_and_query_entity_from_source_side(conn):
    upsert_entity(conn, "Иван", "человек")
    upsert_entity(conn, "Проект Х", "проект")
    add_relationship(conn, "Иван", "Проект Х", "автор_создатель", "руководит проектом", "заметка.md")

    results = query_entity(conn, "Иван")

    assert len(results) == 1
    assert results[0]["entity"] == "Проект Х"
    assert results[0]["category"] == "автор_создатель"
    assert results[0]["description"] == "руководит проектом"
    assert results[0]["source_path"] == "заметка.md"


def test_query_entity_finds_relationship_from_target_side_too(conn):
    upsert_entity(conn, "Иван", "человек")
    upsert_entity(conn, "Проект Х", "проект")
    add_relationship(conn, "Иван", "Проект Х", "автор_создатель", "руководит проектом", "заметка.md")

    results = query_entity(conn, "Проект Х")

    assert len(results) == 1
    assert results[0]["entity"] == "Иван"


def test_delete_relationships_by_source_removes_only_matching_edges(conn):
    upsert_entity(conn, "A", "концепт")
    upsert_entity(conn, "B", "концепт")
    upsert_entity(conn, "C", "концепт")
    add_relationship(conn, "A", "B", "другое", "desc1", "file1.md")
    add_relationship(conn, "A", "C", "другое", "desc2", "file2.md")

    delete_relationships_by_source(conn, "file1.md")

    results_a = query_entity(conn, "A")
    assert len(results_a) == 1
    assert results_a[0]["entity"] == "C"
