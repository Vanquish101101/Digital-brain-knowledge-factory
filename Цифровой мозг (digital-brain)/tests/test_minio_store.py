import uuid

import pytest

from kf.config import load_settings
from kf.store.minio_store import ensure_bucket, file_exists, get_client, upload_file


@pytest.fixture
def client():
    settings = load_settings()
    c = get_client(settings)
    ensure_bucket(c)
    return c


def test_uploaded_file_exists(client, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("привет из теста", encoding="utf-8")
    object_name = f"test/{uuid.uuid4()}.txt"

    upload_file(client, f, object_name)

    assert file_exists(client, object_name) is True


def test_missing_file_does_not_exist(client):
    assert file_exists(client, f"test/{uuid.uuid4()}.txt") is False
