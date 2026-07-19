# Синхронизация удалений (kf.py sync-deletions) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Новая команда `kf.py sync-deletions [--yes]`, которая находит файлы, реально
удалённые из `raw-data-repository/` (отличая их от переименованных по совпадению
sha256-хэша), и по явному подтверждению чистит осиротевшие записи во всех хранилищах:
Postgres, все коллекции Qdrant (по всем профилям эмбеддинга), MinIO, граф знаний, и
физический файл синтезированной заметки на диске.

**Architecture:** Новый модуль `kf/deletion_sync.py` — чистая функция определения
удалено/переименовано (тестируется без БД) + функция реального удаления по всем хранилищам
(тестируется против живого стека, как остальные store-уровневые тесты проекта). Две мелкие
переиспользуемые добавки в существующие `kf/store/postgres.py` и `kf/store/minio_store.py`.
Новая CLI-команда связывает всё воедино, без изменений в существующих командах.

**Tech Stack:** Python 3.12, psycopg (Postgres), qdrant-client, minio, kuzu (граф), click,
pytest.

## Global Constraints

- Различение «удалено» vs «переименовано» — по сравнению sha256-хэша пропавшего пути с
  хэшами всех путей, которые всё ещё существуют на диске. Совпадение = переименование, не
  трогаем.
- Без `--yes` команда только показывает план (что почистит, что пропустит как переименование)
  и не вносит изменений. Только `--yes` реально удаляет.
- Очистка идёт по **всем** профилям эмбеддинга в Qdrant (`kf.embedding_models.EMBEDDING_PROFILES`),
  не только по активному — иначе при возврате на ранее использованную модель удалённый
  контент всплывёт снова.
- Очищается и исходный файл, и его синтезированная заметка (в Postgres/Qdrant/MinIO), плюс
  сам физический файл заметки на диске. Граф знаний чистится только по пути исходного файла
  (сущности всегда извлекаются из исходного текста, не из заметки).
- Сбой очистки в одном хранилище не должен останавливать очистку остальных — каждый шаг в
  своём `try/except` с логом, тот же принцип, что уже применён в `kf/ingest.py`.
- После успешной очистки — запись в `Журнал знаний.md` с действием `очищено_из_базы` для
  каждого очищенного файла.
- Вне рамок этого плана: контентная «уборка» устаревшего/избыточного, но не удалённого
  материала — это отдельный будущий модуль (см. спеку, раздел «Не цели»).

---

### Task 1: Утилиты хранилищ — `list_paths_with_hashes` и `remove_object`

**Files:**
- Modify: `knowledge-factory/kf/store/postgres.py`
- Modify: `knowledge-factory/kf/store/minio_store.py`
- Test: `knowledge-factory/tests/test_postgres_store.py`
- Test: `knowledge-factory/tests/test_minio_store.py`

**Interfaces:**
- Consumes: ничего нового.
- Produces: `list_paths_with_hashes(conn: psycopg.Connection, exclude_prefix: str = "") -> dict[str, str]`
  (путь → sha256, по образцу уже существующего `list_paths`); `remove_object(client: Minio, object_name: str, bucket: str = BUCKET) -> None`.
  Используются в Task 2 (`kf/deletion_sync.py`) и Task 3 (CLI-команда).

- [ ] **Step 1: Write the failing tests**

Добавить в конец `knowledge-factory/tests/test_postgres_store.py`:

```python
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
```

Изменить строку импорта в начале того же файла:

```python
from kf.store.postgres import connect, ensure_schema, needs_ingest, record_ingested, list_paths, path_known
```

на:

```python
from kf.store.postgres import connect, ensure_schema, needs_ingest, record_ingested, list_paths, list_paths_with_hashes, path_known
```

Добавить в конец `knowledge-factory/tests/test_minio_store.py`:

```python
def test_remove_object_deletes_uploaded_file(client, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("привет из теста", encoding="utf-8")
    object_name = f"test/{uuid.uuid4()}.txt"
    upload_file(client, f, object_name)

    remove_object(client, object_name)

    assert file_exists(client, object_name) is False
```

Изменить строку импорта в начале того же файла:

```python
from kf.store.minio_store import ensure_bucket, file_exists, get_client, upload_file
```

на:

```python
from kf.store.minio_store import ensure_bucket, file_exists, get_client, remove_object, upload_file
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_postgres_store.py tests/test_minio_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'list_paths_with_hashes'` и
`ImportError: cannot import name 'remove_object'`.

- [ ] **Step 3: Implement**

В `knowledge-factory/kf/store/postgres.py` добавить в конец файла:

```python
def list_paths_with_hashes(conn: psycopg.Connection, exclude_prefix: str = "") -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT path, sha256 FROM documents")
        rows = cur.fetchall()
    result = {row[0]: row[1] for row in rows}
    if exclude_prefix:
        result = {p: h for p, h in result.items() if not p.startswith(exclude_prefix)}
    return result
```

В `knowledge-factory/kf/store/minio_store.py` добавить в конец файла:

```python
def remove_object(client: Minio, object_name: str, bucket: str = BUCKET) -> None:
    client.remove_object(bucket, object_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_postgres_store.py tests/test_minio_store.py -v`
Expected: PASS — все тесты обоих файлов.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/store/postgres.py" "knowledge-factory/kf/store/minio_store.py" "knowledge-factory/tests/test_postgres_store.py" "knowledge-factory/tests/test_minio_store.py"
git commit -m "Digital brain | Утилиты хранилищ: list_paths_with_hashes + remove_object | V 1.4.58"
```

---

### Task 2: Модуль `kf/deletion_sync.py`

**Files:**
- Create: `knowledge-factory/kf/deletion_sync.py`
- Test: `knowledge-factory/tests/test_deletion_sync.py`

**Interfaces:**
- Consumes: `EMBEDDING_PROFILES` (`kf.embedding_models`, существующий); `delete_by_path`
  (`kf.store.qdrant_store`, существующий); `remove_object` из Task 1
  (`kf.store.minio_store`); `delete_relationships_by_source` (`kf.store.graph_store`,
  существующий); `kf.config.Settings` (существующий тип, поле `synthesis_notes_dir`).
- Produces: `compute_deletion_candidates(known_source_hashes: dict[str, str], seen_source_paths: set[str], all_known_hashes: dict[str, str]) -> tuple[list[str], list[str]]`
  (возвращает `(confirmed_deleted, likely_renamed)`, оба отсортированы);
  `note_rel_key_for(source_rel_key: str, notes_dir_name: str) -> str`;
  `purge_source(source_rel_key: str, settings: Settings, pg_conn, qdrant_client, minio_client, graph_conn) -> None`.
  Используются в Task 3 (CLI-команда).

- [ ] **Step 1: Write the failing tests**

Создать `knowledge-factory/tests/test_deletion_sync.py`:

```python
import uuid
from pathlib import Path

from kf.config import load_settings
from kf.deletion_sync import compute_deletion_candidates, note_rel_key_for, purge_source
from kf.embedding_models import EmbeddingProfile
from kf.store.graph_store import add_relationship, ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.graph_store import query_entity, upsert_entity
from kf.store.minio_store import ensure_bucket, file_exists
from kf.store.minio_store import get_client as get_minio_client
from kf.store.minio_store import upload_file
from kf.store.postgres import connect, ensure_schema, path_known, record_ingested
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import list_paths as qdrant_list_paths
from kf.store.qdrant_store import upsert_chunks


def test_compute_deletion_candidates_flags_missing_path_as_deleted():
    known = {"a.md": "hash-a", "b.md": "hash-b"}
    seen = {"a.md"}
    all_hashes = {"a.md": "hash-a", "b.md": "hash-b"}

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == ["b.md"]
    assert renamed == []


def test_compute_deletion_candidates_detects_rename_by_matching_hash():
    known = {"old.md": "hash-x"}
    seen = {"new.md"}
    all_hashes = {"old.md": "hash-x", "new.md": "hash-x"}

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == []
    assert renamed == ["old.md"]


def test_compute_deletion_candidates_returns_empty_when_nothing_missing():
    known = {"a.md": "hash-a"}
    seen = {"a.md"}
    all_hashes = {"a.md": "hash-a"}

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == []
    assert renamed == []


def test_compute_deletion_candidates_sorts_results():
    known = {"z.md": "h1", "a.md": "h2"}
    seen: set[str] = set()
    all_hashes = known

    confirmed, renamed = compute_deletion_candidates(known, seen, all_hashes)

    assert confirmed == ["a.md", "z.md"]


def test_note_rel_key_for_builds_expected_path():
    assert (
        note_rel_key_for("003 Знания/файл.md", "synthesized-notes")
        == "synthesized-notes/003 Знания/файл.md.md"
    )


def test_purge_source_removes_from_all_stores(tmp_path, monkeypatch):
    settings = load_settings()
    settings.synthesis_notes_dir = str(tmp_path / "notes")
    Path(settings.synthesis_notes_dir).mkdir(parents=True, exist_ok=True)

    source_rel_key = "test-purge/файл.md"
    notes_dir_name = Path(settings.synthesis_notes_dir).name
    note_rel_key = note_rel_key_for(source_rel_key, notes_dir_name)

    # Фейковые профили с выделенными тестовыми коллекциями — НЕ трогаем реальную
    # продакшен-коллекцию "knowledge". Два профиля — чтобы доказать, что purge_source
    # реально чистит каждую коллекцию по очереди, а не только одну.
    test_collections = ["kf_test_deletion_sync_a", "kf_test_deletion_sync_b"]
    fake_profiles = {
        name: EmbeddingProfile(name=name, provider="local", model_id="m", dimension=8, collection=collection)
        for name, collection in zip(("test-a", "test-b"), test_collections)
    }
    monkeypatch.setattr("kf.deletion_sync.EMBEDDING_PROFILES", fake_profiles)

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    record_ingested(pg_conn, source_rel_key, "hash-x")
    record_ingested(pg_conn, note_rel_key, "hash-y")

    qdrant_client = get_qdrant_client(settings)
    vector = [0.0] * 8
    try:
        for collection in test_collections:
            ensure_collection(qdrant_client, collection, vector_size=8)
            upsert_chunks(
                qdrant_client,
                collection,
                [{"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": source_rel_key, "chunk_index": 0, "text": "x"}}],
            )

        minio_client = get_minio_client(settings)
        ensure_bucket(minio_client)
        raw_file = tmp_path / "raw.md"
        raw_file.write_text("тест", encoding="utf-8")
        upload_file(minio_client, raw_file, source_rel_key)

        graph_conn = get_graph_connection(settings)
        ensure_graph_schema(graph_conn)
        upsert_entity(graph_conn, "ТестоваяСущность", "концепт")
        upsert_entity(graph_conn, "ДругаяСущность", "концепт")
        add_relationship(graph_conn, "ТестоваяСущность", "ДругаяСущность", "связано_с_темой", "тест", source_rel_key)

        note_path = Path(settings.synthesis_notes_dir) / f"{source_rel_key}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("заметка", encoding="utf-8")

        purge_source(source_rel_key, settings, pg_conn, qdrant_client, minio_client, graph_conn)

        assert path_known(pg_conn, source_rel_key) is False
        assert path_known(pg_conn, note_rel_key) is False
        for collection in test_collections:
            assert qdrant_list_paths(qdrant_client, collection) == set()
        assert file_exists(minio_client, source_rel_key) is False
        assert file_exists(minio_client, note_rel_key) is False
        assert query_entity(graph_conn, "ТестоваяСущность") == []
        assert not note_path.exists()
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE path IN (%s, %s)", (source_rel_key, note_rel_key))
        pg_conn.commit()
        for collection in test_collections:
            try:
                qdrant_client.delete_collection(collection)
            except Exception:
                pass
```

Важно: `EMBEDDING_PROFILES` подменяется через `monkeypatch.setattr("kf.deletion_sync.EMBEDDING_PROFILES", ...)` —
именно на объект в модуле `kf.deletion_sync` (куда он импортирован), а не в
`kf.embedding_models` — иначе подмена не подействует на уже импортированную внутри
`deletion_sync.py` ссылку.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_deletion_sync.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.deletion_sync'`.

- [ ] **Step 3: Implement**

Создать `knowledge-factory/kf/deletion_sync.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_deletion_sync.py -v`
Expected: PASS (6/6). Если Docker/Qdrant/Postgres/MinIO нездоровы на момент запуска —
ошибка будет сетевой (`connection failed`/`Server disconnected`), не логической; в этом
случае проверить `docker ps`, при необходимости `docker restart <контейнер>` и повторить —
известная особенность стека на Hyper-V в этом проекте, не баг кода.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/deletion_sync.py" "knowledge-factory/tests/test_deletion_sync.py"
git commit -m "Digital brain | Модуль kf/deletion_sync.py: определение и выполнение очистки удалённых файлов | V 1.4.59"
```

---

### Task 3: Команда `kf.py sync-deletions [--yes]`

**Files:**
- Modify: `knowledge-factory/kf/cli.py`
- Test: `knowledge-factory/tests/test_cli.py`

**Interfaces:**
- Consumes: `compute_deletion_candidates`, `purge_source` из Task 2 (`kf.deletion_sync`);
  `list_paths_with_hashes` из Task 1 (`kf.store.postgres`); `should_index`
  (`kf.scope`, существующий); `format_entry`, `append_entries` (`kf.journal`, существующие).
- Produces: CLI-команда `kf.py sync-deletions [--yes]`.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_cli.py`:

```python
def test_sync_deletions_dry_run_shows_plan_without_purging(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.DEFAULT_SOURCE", tmp_path)
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "kf.cli.list_paths_with_hashes",
        lambda pg_conn, exclude_prefix="": {"a.md": "hash-a", "b.md": "hash-b"},
    )
    purge_calls = []
    monkeypatch.setattr("kf.cli.purge_source", lambda *a, **kw: purge_calls.append(a))

    (tmp_path / "a.md").write_text("x", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["sync-deletions"])

    assert result.exit_code == 0
    assert "b.md" in result.output
    assert "К очистке" in result.output
    assert "предварительный просмотр" in result.output
    assert purge_calls == []


def test_sync_deletions_with_yes_calls_purge_and_writes_journal(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.DEFAULT_SOURCE", tmp_path)
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "kf.cli.list_paths_with_hashes",
        lambda pg_conn, exclude_prefix="": {"a.md": "hash-a", "b.md": "hash-b"},
    )
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)
    monkeypatch.setattr("kf.cli.get_minio_client", lambda s: None)
    monkeypatch.setattr("kf.cli.get_graph_connection", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_graph_schema", lambda conn: None)
    purge_calls = []
    monkeypatch.setattr("kf.cli.purge_source", lambda *a, **kw: purge_calls.append(a[0]))

    (tmp_path / "a.md").write_text("x", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["sync-deletions", "--yes"])

    assert result.exit_code == 0
    assert purge_calls == ["b.md"]
    assert "Очищено: 1" in result.output
    journal = tmp_path.parent / "Журнал знаний.md"
    assert journal.exists()
    assert "очищено_из_базы" in journal.read_text(encoding="utf-8")
    journal.unlink()


def test_sync_deletions_reports_nothing_to_clean(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.DEFAULT_SOURCE", tmp_path)
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr(
        "kf.cli.list_paths_with_hashes",
        lambda pg_conn, exclude_prefix="": {"a.md": "hash-a"},
    )

    (tmp_path / "a.md").write_text("x", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["sync-deletions"])

    assert result.exit_code == 0
    assert "Нечего чистить" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_cli.py -k sync_deletions -v`
Expected: FAIL — `Error: No such command 'sync-deletions'`.

- [ ] **Step 3: Implement**

В `knowledge-factory/kf/cli.py` заменить блок импортов в начале файла:

```python
from pathlib import Path

import click

from kf.api import ask_question, get_stats, graph_search, open_session, semantic_search
from kf.config import load_settings
from kf.embedding_models import EMBEDDING_PROFILES, get_profile
from kf.embedding_state import get_active_profile_name, set_active_profile_name
from kf.embedding_sync import sync_missing_paths
from kf.embeddings import get_embedder_for_profile
from kf.ingest import IngestDeps, ingest_directory
from kf.journal import detect_deleted
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.minio_store import ensure_bucket
from kf.store.minio_store import get_client as get_minio_client
from kf.store.postgres import connect, ensure_schema, list_paths
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import list_paths as qdrant_list_paths
from kf.web_extract import derive_filename, extract_from_url

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "raw-data-repository"
```

на:

```python
from datetime import date
from pathlib import Path

import click

from kf.api import ask_question, get_stats, graph_search, open_session, semantic_search
from kf.config import load_settings
from kf.deletion_sync import compute_deletion_candidates, purge_source
from kf.embedding_models import EMBEDDING_PROFILES, get_profile
from kf.embedding_state import get_active_profile_name, set_active_profile_name
from kf.embedding_sync import sync_missing_paths
from kf.embeddings import get_embedder_for_profile
from kf.ingest import IngestDeps, ingest_directory
from kf.journal import append_entries, detect_deleted, format_entry
from kf.scope import should_index
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.minio_store import ensure_bucket
from kf.store.minio_store import get_client as get_minio_client
from kf.store.postgres import connect, ensure_schema, list_paths, list_paths_with_hashes
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import list_paths as qdrant_list_paths
from kf.web_extract import derive_filename, extract_from_url

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "raw-data-repository"
```

Добавить новую команду в конец `knowledge-factory/kf/cli.py`, перед `if __name__ == "__main__":`:

```python
@cli.command(name="sync-deletions")
@click.option("--yes", is_flag=True, default=False, help="Реально выполнить очистку (без флага — только план).")
def sync_deletions(yes: bool):
    """Найти файлы, реально удалённые из vault, и (по подтверждению) очистить осиротевшие записи."""
    settings = load_settings()
    pg_conn = connect(settings)
    ensure_schema(pg_conn)

    notes_dir_name = Path(settings.synthesis_notes_dir).name
    notes_prefix = f"{notes_dir_name}/"

    all_hashes = list_paths_with_hashes(pg_conn)
    known_source_hashes = list_paths_with_hashes(pg_conn, exclude_prefix=notes_prefix)

    seen_source_paths = {
        path.relative_to(DEFAULT_SOURCE).as_posix()
        for path in sorted(DEFAULT_SOURCE.rglob("*"))
        if path.is_file() and should_index(path)
    }

    confirmed, renamed = compute_deletion_candidates(known_source_hashes, seen_source_paths, all_hashes)

    if renamed:
        click.echo(f"Похоже на переименование/дубликат (не трогаем): {len(renamed)}")
        for p in renamed:
            click.echo(f"  - {p}")

    if not confirmed:
        click.echo("Нечего чистить — осиротевших записей не найдено.")
        return

    click.echo(f"К очистке: {len(confirmed)} файл(ов)")
    for p in confirmed:
        click.echo(f"  - {p}")

    if not yes:
        click.echo("\nЭто был предварительный просмотр. Запустите с --yes, чтобы реально очистить.")
        return

    qdrant_client = get_qdrant_client(settings)
    minio_client = get_minio_client(settings)
    graph_conn = get_graph_connection(settings)
    ensure_graph_schema(graph_conn)

    journal_entries = []
    today = date.today().isoformat()
    for p in confirmed:
        purge_source(p, settings, pg_conn, qdrant_client, minio_client, graph_conn)
        section = p.split("/", 1)[0]
        journal_entries.append(format_entry("очищено_из_базы", p, section, "", today))

    append_entries(journal_entries, DEFAULT_SOURCE.parent / "Журнал знаний.md")
    click.echo(f"Готово. Очищено: {len(confirmed)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_cli.py -v`
Expected: PASS — все тесты файла, включая три новых.

Затем прогнать полный набор тестов:

Run: `cd "knowledge-factory" && uv run python -m pytest -q`
Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/cli.py" "knowledge-factory/tests/test_cli.py"
git commit -m "Digital brain | Команда kf.py sync-deletions | V 1.4.60"
```

---

### Task 4: Документация

**Files:**
- Modify: `E:\Digital brain\карта_бз.md`
- Modify: `knowledge-factory/README.md`

Задача без тестов (документация) — TDD не применяется, отдельный шаг с коммитом.

- [ ] **Step 1: Обновить `E:\Digital brain\карта_бз.md`**

Найти раздел:

```markdown
## Журнал знаний (Журнал знаний.md)

Автоматический лог каждого добавления/изменения/удаления файла в `raw-data-repository/`.
Пополняется сам при каждом запуске `kf.py ingest` (в т.ч. если файлы менялись вручную,
напрямую в Obsidian, без участия ассистента) — редактировать вручную не нужно и не имеет
смысла, правки перезапишутся при следующем `ingest`. При обнаружении удалённых файлов
запись попадает в журнал, но векторы/строки в Qdrant/Postgres/MinIO не удаляются
автоматически — это отдельное решение, принимается пользователем по запросу ассистента.
```

Заменить на:

```markdown
## Журнал знаний (Журнал знаний.md)

Автоматический лог каждого добавления/изменения/удаления файла в `raw-data-repository/`.
Пополняется сам при каждом запуске `kf.py ingest` (в т.ч. если файлы менялись вручную,
напрямую в Obsidian, без участия ассистента) — редактировать вручную не нужно и не имеет
смысла, правки перезапишутся при следующем `ingest`. При обнаружении удалённых файлов
запись попадает в журнал, но векторы/строки в Qdrant/Postgres/MinIO не удаляются
автоматически при обычном `ingest`.

Для реальной очистки — `kf.py sync-deletions` (сначала без флагов — покажет план и отличит
реальное удаление от переименования по сравнению хэша содержимого; `--yes` — выполняет).
Чистит и исходный файл, и его синтезированную заметку, во всех коллекциях Qdrant по всем
профилям эмбеддинга, в MinIO и в графе знаний.
```

- [ ] **Step 2: Обновить `knowledge-factory/README.md`**

Найти блок:

```markdown
uv run python kf.py embedding-model sync     # досчитать недостающие эмбеддинги для активной модели
```
```

Заменить на:

```markdown
uv run python kf.py embedding-model sync     # досчитать недостающие эмбеддинги для активной модели
uv run python kf.py sync-deletions [--yes]   # очистить записи для файлов, реально удалённых из vault
```
```

- [ ] **Step 3: Commit**

```bash
git add "карта_бз.md" "knowledge-factory/README.md"
git commit -m "Digital brain | Документация: kf.py sync-deletions | V 1.4.61"
```
