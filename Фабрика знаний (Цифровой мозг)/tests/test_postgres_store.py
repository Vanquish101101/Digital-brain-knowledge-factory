import pytest

from kf.config import load_settings
from kf.store.postgres import connect, ensure_schema, needs_ingest, record_ingested


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
