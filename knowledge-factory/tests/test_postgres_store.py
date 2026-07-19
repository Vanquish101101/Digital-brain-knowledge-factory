import pytest

from kf.config import load_settings
from kf.store.postgres import connect, ensure_schema, needs_ingest, record_ingested, list_paths, list_paths_with_hashes, path_known


@pytest.fixture
def conn():
    settings = load_settings()
    connection = connect(settings)
    ensure_schema(connection)
    with connection.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE 'test://%'")
    connection.commit()
    yield connection
    with connection.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE 'test://%'")
    connection.commit()
    connection.close()


def test_unseen_path_needs_ingest(conn):
    assert needs_ingest(conn, "test://new-file.md", "hash-a") is True


def test_recorded_path_with_same_hash_does_not_need_ingest(conn):
    record_ingested(conn, "test://note.md", "hash-a")

    assert needs_ingest(conn, "test://note.md", "hash-a") is False


def test_recorded_path_with_changed_hash_needs_ingest(conn):
    record_ingested(conn, "test://note.md", "hash-a")

    assert needs_ingest(conn, "test://note.md", "hash-b") is True


def test_record_ingested_is_idempotent(conn):
    record_ingested(conn, "test://note.md", "hash-a")
    record_ingested(conn, "test://note.md", "hash-b")

    assert needs_ingest(conn, "test://note.md", "hash-b") is False


def test_path_known_false_for_unseen_path(conn):
    assert path_known(conn, "test://unknown.md") is False


def test_path_known_true_after_record_ingested(conn):
    record_ingested(conn, "test://known.md", "hash-a")

    assert path_known(conn, "test://known.md") is True


def test_list_paths_returns_all_recorded_paths(conn):
    record_ingested(conn, "test://a.md", "hash-a")
    record_ingested(conn, "test://b.md", "hash-b")

    paths = list_paths(conn)

    assert "test://a.md" in paths
    assert "test://b.md" in paths


def test_list_paths_excludes_prefix(conn):
    record_ingested(conn, "test://a.md", "hash-a")
    record_ingested(conn, "test://notes/a.md.md", "hash-b")

    paths = list_paths(conn, exclude_prefix="test://notes/")

    assert "test://a.md" in paths
    assert "test://notes/a.md.md" not in paths


def test_list_paths_with_hashes_returns_hash_per_path(conn):
    record_ingested(conn, "test://a.md", "hash-a")
    record_ingested(conn, "test://b.md", "hash-b")

    result = list_paths_with_hashes(conn)

    assert result["test://a.md"] == "hash-a"
    assert result["test://b.md"] == "hash-b"


def test_list_paths_with_hashes_excludes_prefix(conn):
    record_ingested(conn, "test://a.md", "hash-a")
    record_ingested(conn, "test://notes/a.md.md", "hash-b")

    result = list_paths_with_hashes(conn, exclude_prefix="test://notes/")

    assert "test://a.md" in result
    assert "test://notes/a.md.md" not in result
