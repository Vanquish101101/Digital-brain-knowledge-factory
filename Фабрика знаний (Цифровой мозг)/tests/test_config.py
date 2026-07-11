from kf.config import load_settings


def test_finds_env_file_regardless_of_current_working_directory(monkeypatch, tmp_path):
    for key in [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.postgres_user == "kf"
    assert settings.postgres_db == "knowledge_factory"


def test_loads_postgres_settings_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("MINIO_ROOT_USER", "minioadmin")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "miniosecret")

    settings = load_settings()

    assert settings.postgres_user == "testuser"
    assert settings.postgres_password == "testpass"
    assert settings.postgres_db == "testdb"
    assert settings.minio_access_key == "minioadmin"
    assert settings.minio_secret_key == "miniosecret"


def test_defaults_host_and_ports(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("MINIO_ROOT_USER", "m")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "s")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    settings = load_settings()

    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.qdrant_url == "http://localhost:6333"
