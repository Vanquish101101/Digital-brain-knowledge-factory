# Журнал знаний + регламент маршрутизации — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Автоматический журнал добавлений/изменений/удалений в `raw-data-repository/`
(`Журнал знаний.md` в корне проекта), пополняемый при каждом запуске `kf.py ingest`,
плюс расширенный регламент маршрутизации по теме материала в `карта_бз.md`.

**Architecture:** Новый чистый модуль `kf/journal.py` (форматирование строк журнала,
детекция удалений как разница множеств путей) + новые вспомогательные функции в
`kf/store/postgres.py` (`path_known`, `list_paths`) + интеграция в `kf/ingest.py::ingest_directory`
(сбор записей по ходу скана, детекция удалений после скана, один вызов записи в файл в конце) +
расширение вывода `kf.py ingest` в `kf/cli.py` + документация (`карта_бз.md`, `CLAUDE.md`,
`AGENTS.md`).

**Tech Stack:** Python 3.12, psycopg (Postgres), pytest, click. Тот же стек, что и весь
остальной `kf/`.

## Global Constraints

- Журнал `Журнал знаний.md` — в корне проекта (`source_dir.parent`), НЕ внутри
  `raw-data-repository/` — не должен попадать в собственную индексацию.
- Обнаружение удалений НЕ удаляет векторы/строки из Qdrant/Postgres/MinIO автоматически —
  только логирует факт в журнал. Решение об очистке базы — открытый вопрос, отложен
  пользователем, реализуется отдельно.
- Журнал пополняется только при запуске `kf.py ingest` — не в реальном времени, без
  файлового watcher-а.
- Существующие 23 документа в базе на момент написания этого плана в журнал задним числом
  не заносятся — журнал стартует с нуля со следующего запуска `ingest` после реализации.
- Описание для записи о добавлении/изменении берётся из уже сгенерированной синтез-заметки
  (без дополнительного вызова LLM) — правило извлечения: первая непустая строка текста
  заметки, без ведущей нумерации/маркдаун-разметки, обрезанная до 150 символов.
- Регламент маршрутизации по теме и по типу материала — единственный источник правды
  находится в таблице "Куда класть новое" в `карта_бз.md`; `CLAUDE.md`/`AGENTS.md` на неё
  ссылаются, не дублируют содержимое.
- Не менять существующий контракт `needs_ingest`/`record_ingested` — добавлять новые
  небольшие функции рядом, а не переделывать их сигнатуры (у них уже есть собственные тесты
  в `tests/test_postgres_store.py`).

---

### Task 1: `path_known` и `list_paths` в `kf/store/postgres.py`

**Files:**
- Modify: `knowledge-factory/kf/store/postgres.py`
- Test: `knowledge-factory/tests/test_postgres_store.py`

**Interfaces:**
- Consumes: ничего нового, использует уже существующий `documents` (path TEXT UNIQUE,
  sha256 TEXT) и `connect`/`ensure_schema`/`record_ingested` из этого же файла.
- Produces: `path_known(conn: psycopg.Connection, path: str) -> bool`,
  `list_paths(conn: psycopg.Connection, exclude_prefix: str = "") -> set[str]` — используются
  в Task 3 (`kf/ingest.py`).

- [ ] **Step 1: Write the failing tests**

Добавить в конец `knowledge-factory/tests/test_postgres_store.py`:

```python
from kf.store.postgres import list_paths, path_known


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
```

(Импорт `record_ingested` уже есть в верхней части файла — импорт `list_paths, path_known`
добавляется отдельной строкой, как показано выше.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run pytest tests/test_postgres_store.py -v`
Expected: FAIL с `ImportError: cannot import name 'list_paths'` (и `path_known`).

- [ ] **Step 3: Implement**

В конец `knowledge-factory/kf/store/postgres.py` добавить:

```python
def path_known(conn: psycopg.Connection, path: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE path = %s", (path,))
        return cur.fetchone() is not None


def list_paths(conn: psycopg.Connection, exclude_prefix: str = "") -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT path FROM documents")
        rows = cur.fetchall()
    paths = {row[0] for row in rows}
    if exclude_prefix:
        paths = {p for p in paths if not p.startswith(exclude_prefix)}
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run pytest tests/test_postgres_store.py -v`
Expected: PASS (все тесты в файле, включая уже существующие).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/store/postgres.py" "knowledge-factory/tests/test_postgres_store.py"
git commit -m "Digital brain | path_known/list_paths в postgres store для журнала знаний | V 1.4.27"
```

---

### Task 2: Модуль `kf/journal.py`

**Files:**
- Create: `knowledge-factory/kf/journal.py`
- Test: `knowledge-factory/tests/test_journal.py`

**Interfaces:**
- Consumes: ничего (чистые функции, только `pathlib.Path`).
- Produces: `extract_description(note_text: str) -> str`,
  `format_entry(action: str, path: str, section: str, description: str, date: str) -> str`,
  `detect_deleted(known_paths: set[str], seen_paths: set[str]) -> set[str]`,
  `append_entries(entries: list[str], path: Path) -> None` — все используются в Task 3.

- [ ] **Step 1: Write the failing tests**

Создать `knowledge-factory/tests/test_journal.py`:

```python
from kf.journal import append_entries, detect_deleted, extract_description, format_entry


def test_extract_description_strips_leading_numbering():
    note = "1. Материал про Docker и контейнеризацию.\n2. Ключевые идеи..."

    assert extract_description(note) == "Материал про Docker и контейнеризацию."


def test_extract_description_strips_leading_dash():
    note = "- Курс по 3D-моделированию в Blender.\nПодробности дальше."

    assert extract_description(note) == "Курс по 3D-моделированию в Blender."


def test_extract_description_truncates_long_first_line():
    note = "x" * 200

    assert extract_description(note) == "x" * 150


def test_extract_description_empty_for_blank_note():
    assert extract_description("") == ""
    assert extract_description("   \n  ") == ""


def test_format_entry_with_description():
    line = format_entry(
        "добавлено", "005 Ресурсы/курс.pdf", "005 Ресурсы", "О курсе по 3D", "2026-07-16"
    )

    assert line == "- 2026-07-16 | добавлено | 005 Ресурсы/курс.pdf | 005 Ресурсы | О курсе по 3D"


def test_format_entry_without_description():
    line = format_entry("удалено", "003 Знания/старое.md", "003 Знания", "", "2026-07-16")

    assert line == "- 2026-07-16 | удалено | 003 Знания/старое.md | 003 Знания | (без описания)"


def test_detect_deleted_finds_paths_missing_from_seen():
    known = {"a.md", "b.md", "c.md"}
    seen = {"a.md", "c.md"}

    assert detect_deleted(known, seen) == {"b.md"}


def test_detect_deleted_empty_when_nothing_missing():
    known = {"a.md"}
    seen = {"a.md"}

    assert detect_deleted(known, seen) == set()


def test_append_entries_creates_file_with_header(tmp_path):
    journal = tmp_path / "Журнал знаний.md"

    append_entries(["- entry one"], journal)

    content = journal.read_text(encoding="utf-8")
    assert "# Журнал знаний" in content
    assert "- entry one" in content


def test_append_entries_appends_to_existing_file(tmp_path):
    journal = tmp_path / "Журнал знаний.md"
    append_entries(["- entry one"], journal)

    append_entries(["- entry two"], journal)

    content = journal.read_text(encoding="utf-8")
    assert "- entry one" in content
    assert "- entry two" in content


def test_append_entries_noop_for_empty_list(tmp_path):
    journal = tmp_path / "Журнал знаний.md"

    append_entries([], journal)

    assert not journal.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run pytest tests/test_journal.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.journal'`.

- [ ] **Step 3: Implement**

Создать `knowledge-factory/kf/journal.py`:

```python
from pathlib import Path

JOURNAL_HEADER = (
    "# Журнал знаний\n\n"
    "Автоматический лог добавлений, изменений и удалений в raw-data-repository/, "
    "пополняется при каждом запуске kf.py ingest.\n\n"
)

_STRIP_CHARS = "0123456789.-*# \t"


def extract_description(note_text: str) -> str:
    stripped = note_text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    cleaned = first_line.lstrip(_STRIP_CHARS)
    return cleaned[:150]


def format_entry(action: str, path: str, section: str, description: str, date: str) -> str:
    desc = description if description else "(без описания)"
    return f"- {date} | {action} | {path} | {section} | {desc}"


def detect_deleted(known_paths: set[str], seen_paths: set[str]) -> set[str]:
    return known_paths - seen_paths


def append_entries(entries: list[str], path: Path) -> None:
    if not entries:
        return
    if not path.exists():
        path.write_text(JOURNAL_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run pytest tests/test_journal.py -v`
Expected: PASS (11/11).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/journal.py" "knowledge-factory/tests/test_journal.py"
git commit -m "Digital brain | Модуль kf/journal.py: форматирование и детекция удалений | V 1.4.28"
```

---

### Task 3: Интеграция журнала в `kf/ingest.py`

**Files:**
- Modify: `knowledge-factory/kf/ingest.py` (весь файл)
- Test: `knowledge-factory/tests/test_ingest.py`

**Interfaces:**
- Consumes: `path_known`, `list_paths` из Task 1 (`kf.store.postgres`);
  `extract_description`, `format_entry`, `detect_deleted`, `append_entries` из Task 2
  (`kf.journal`).
- Produces: `IngestStats` получает новые поля `journal_entries_written: int = 0` и
  `deleted_detected: int = 0`; `_synthesize_and_index_note` теперь возвращает
  `str | None` (текст заметки или `None` при неудаче синтеза) вместо `None` всегда —
  используется в Task 4 (`kf/cli.py`) косвенно через новые поля `IngestStats`.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_ingest.py`:

```python
def test_journal_logs_added_then_changed_actions(tmp_path, deps):
    f = tmp_path / "note-journal.md"
    f.write_text("Первая версия заметки.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.write_text("Изменённая версия заметки.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    journal_path = tmp_path.parent / "Журнал знаний.md"
    content = journal_path.read_text(encoding="utf-8")
    assert "добавлено | note-journal.md" in content
    assert "изменено | note-journal.md" in content


def test_deleted_file_is_logged_in_journal(tmp_path, deps):
    f = tmp_path / "note-to-delete.md"
    f.write_text("Заметка, которую скоро удалим.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.unlink()
    stats = ingest_directory(tmp_path, deps)

    journal_path = tmp_path.parent / "Журнал знаний.md"
    content = journal_path.read_text(encoding="utf-8")
    assert "удалено | note-to-delete.md" in content
    assert stats.deleted_detected >= 1


def test_journal_entry_uses_synthesis_note_first_line_as_description(tmp_path, deps, monkeypatch):
    monkeypatch.setattr(
        "kf.ingest.synthesize_note",
        lambda settings, text, source_path: "1. Заметка о рецепте борща.\n2. Ключевые идеи...",
    )
    (tmp_path / "note-desc.md").write_text("Заметка про борщ.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    journal_path = tmp_path.parent / "Журнал знаний.md"
    content = journal_path.read_text(encoding="utf-8")
    assert "Заметка о рецепте борща." in content
```

**Важно:** `deps` fixture уже подменяет `kf.ingest.synthesize_note` через `monkeypatch.setattr`
(см. верх `test_ingest.py`) — третий тест выше переопределяет её ещё раз локально, это
штатно (`monkeypatch` в `pytest` поддерживает переопределение внутри теста).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run pytest tests/test_ingest.py -k journal -v`
Expected: FAIL — `AttributeError: 'IngestStats' object has no attribute 'deleted_detected'`
(и/или журнал-файл не создаётся, т.к. логики пока нет).

- [ ] **Step 3: Implement**

Заменить содержимое `knowledge-factory/kf/ingest.py` целиком на:

```python
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embeddings import embed
from kf.extract import extract_text
from kf.hashing import sha256_of_file
from kf.journal import append_entries, detect_deleted, extract_description, format_entry
from kf.scope import should_index
from kf.store.minio_store import upload_file
from kf.store.postgres import list_paths, needs_ingest, path_known, record_ingested
from kf.store.qdrant_store import upsert_chunks
from kf.synthesize import synthesize_note

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
    notes_synthesized: int = 0
    notes_failed: int = 0
    journal_entries_written: int = 0
    deleted_detected: int = 0


def _point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}:{chunk_index}"))


def _store_text(text: str, rel_key: str, path: Path, deps: IngestDeps, stats: IngestStats) -> None:
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


def _synthesize_and_index_note(
    text: str, rel_key: str, deps: IngestDeps, stats: IngestStats
) -> str | None:
    try:
        note_text = synthesize_note(deps.settings, text, rel_key)
    except Exception as exc:
        print(f"[ingest] синтез не удался для {rel_key}: {exc}")
        stats.notes_failed += 1
        return None

    notes_dir = Path(deps.settings.synthesis_notes_dir)
    note_path = notes_dir / f"{rel_key}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    stats.notes_synthesized += 1

    note_rel_key = f"{notes_dir.name}/{rel_key}.md"
    note_hash = sha256_of_file(note_path)
    if needs_ingest(deps.pg_conn, note_rel_key, note_hash):
        _store_text(note_text, note_rel_key, note_path, deps, stats)
        record_ingested(deps.pg_conn, note_rel_key, note_hash)
        stats.files_ingested += 1

    return note_text


def ingest_directory(source_dir: Path, deps: IngestDeps) -> IngestStats:
    stats = IngestStats()
    notes_dir = Path(deps.settings.synthesis_notes_dir).resolve()
    journal_entries: list[str] = []
    seen_paths: set[str] = set()
    today = date.today().isoformat()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or not should_index(path):
            continue

        rel_key = path.relative_to(source_dir).as_posix()
        stats.files_scanned += 1
        seen_paths.add(rel_key)

        file_hash = sha256_of_file(path)
        is_new = not path_known(deps.pg_conn, rel_key)
        if not needs_ingest(deps.pg_conn, rel_key, file_hash):
            stats.files_skipped += 1
            continue

        try:
            text = extract_text(path, deps.settings)
        except Exception as exc:
            print(f"[ingest] пропускаю {rel_key}: {exc}")
            stats.files_failed += 1
            continue

        _store_text(text, rel_key, path, deps, stats)
        record_ingested(deps.pg_conn, rel_key, file_hash)
        stats.files_ingested += 1

        is_note = notes_dir in path.resolve().parents
        note_text = None
        if not is_note:
            note_text = _synthesize_and_index_note(text, rel_key, deps, stats)

        section = rel_key.split("/", 1)[0]
        description = extract_description(note_text) if note_text else ""
        action = "добавлено" if is_new else "изменено"
        journal_entries.append(format_entry(action, rel_key, section, description, today))

    notes_prefix = f"{notes_dir.name}/"
    known_paths = list_paths(deps.pg_conn, exclude_prefix=notes_prefix)
    deleted = detect_deleted(known_paths, seen_paths)
    for deleted_path in sorted(deleted):
        section = deleted_path.split("/", 1)[0]
        journal_entries.append(format_entry("удалено", deleted_path, section, "", today))
    stats.deleted_detected = len(deleted)
    stats.journal_entries_written = len(journal_entries)

    append_entries(journal_entries, source_dir.parent / "Журнал знаний.md")

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run pytest tests/test_ingest.py -v`
Expected: PASS (все тесты файла, включая три новых и все ранее существовавшие).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/ingest.py" "knowledge-factory/tests/test_ingest.py"
git commit -m "Digital brain | Журнал знаний интегрирован в kf.py ingest (добавления/изменения/удаления) | V 1.4.29"
```

---

### Task 4: Вывод в `kf/cli.py`

**Files:**
- Modify: `knowledge-factory/kf/cli.py:47-55`
- Test: `knowledge-factory/tests/test_cli.py`

**Interfaces:**
- Consumes: `IngestStats.journal_entries_written`, `IngestStats.deleted_detected` из Task 3.
- Produces: ничего нового для других задач — терминальная задача плана перед документацией.

- [ ] **Step 1: Write the failing test**

Изменить существующий `test_ingest_reports_summary` в `knowledge-factory/tests/test_cli.py`,
добавив после существующих assert-ов:

```python
    assert "записей в журнале знаний" in result.output
```

(Полный файл после правки — все существующие строки без изменений, плюс эта одна строка
сразу после `assert "заметок синтезировано: 1" in result.output`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "knowledge-factory" && uv run pytest tests/test_cli.py::test_ingest_reports_summary -v`
Expected: FAIL — `assert "записей в журнале знаний" in result.output` не находит строку.

- [ ] **Step 3: Implement**

В `knowledge-factory/kf/cli.py` заменить блок (строки 47-55):

```python
    click.echo(
        f"Готово. просканировано: {stats.files_scanned}, "
        f"проиндексировано: {stats.files_ingested}, "
        f"пропущено (без изменений): {stats.files_skipped}, "
        f"ошибок: {stats.files_failed}, "
        f"чанков записано: {stats.chunks_written}, "
        f"заметок синтезировано: {stats.notes_synthesized}, "
        f"ошибок синтеза: {stats.notes_failed}"
    )
```

на:

```python
    click.echo(
        f"Готово. просканировано: {stats.files_scanned}, "
        f"проиндексировано: {stats.files_ingested}, "
        f"пропущено (без изменений): {stats.files_skipped}, "
        f"ошибок: {stats.files_failed}, "
        f"чанков записано: {stats.chunks_written}, "
        f"заметок синтезировано: {stats.notes_synthesized}, "
        f"ошибок синтеза: {stats.notes_failed}, "
        f"записей в журнале знаний: {stats.journal_entries_written}"
    )
    if stats.deleted_detected:
        click.echo(
            f"⚠ Обнаружены удалённые файлы: {stats.deleted_detected}. "
            f"Проверьте 'Журнал знаний.md' — решение об очистке базы принимается отдельно."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run pytest tests/test_cli.py -v`
Expected: PASS.

Затем прогнать полный набор тестов проекта:

Run: `cd "knowledge-factory" && uv run pytest -q`
Expected: все тесты PASS (базовое число + 4 новых из Task 1 + 11 новых из Task 2 + 3 новых
из Task 3 = существовавшие 76 плюс 18 новых).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/cli.py" "knowledge-factory/tests/test_cli.py"
git commit -m "Digital brain | kf.py ingest сообщает о записях в журнале знаний и удалениях | V 1.4.30"
```

---

### Task 5: Регламент маршрутизации в документации

**Files:**
- Modify: `карта_бз.md`
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `knowledge-factory/README.md`

Задача без тестов (документация) — TDD не применяется, но задача остаётся отдельным шагом
с собственным коммитом.

- [ ] **Step 1: Расширить таблицу "Куда класть новое" в `карта_бз.md`**

Найти в `карта_бз.md` блок:

```markdown
## Куда класть новое (быстрый ориентир)

| Тип материала | Куда |
|---|---|
| Сырое, не разобрано | `raw-data-repository/001 Входящие (Сырые данные)/` |
| Про себя, ценности, стиль мышления | `002 Мышление (Карта личности)/` |
| Конспект, модель, наблюдение | `003 Знания/` |
| Софт/промпт/скилл, которым реально пользуюсь | `004 Инструменты/` |
| Книга, курс, статья, референс | `005 Ресурсы/` |
| Конкретное действие / цель | `006 Задачи (Цели)/` |
| Активный или будущий проект | `007 Проекты/` |
| Хобби без цели делать проект | `008 Интересы (Увлечения)/` |
| Завершено / больше не актуально | `010 Архив (Неиспользуемое)/` |
| Настройки/архитектура самого проекта | `project-config/` |
```

Заменить на:

```markdown
## Куда класть новое (быстрый ориентир)

**Тема важнее формата файла.** Сначала проверь, относится ли материал к одной из уже
существующих тем ниже — если да, кладём туда независимо от того, PDF это, видео, ссылка,
аудио или текст. Если тема ни с чем не совпадает — используем общее правило по типу
материала.

### По теме (если материал явно про одно из направлений)

| Тема | Куда |
|---|---|
| Веб-дизайн | `007 Проекты/001 Веб Дизайн Web Design/` |
| Видео-дизайн | `007 Проекты/002 Видео Дизайн Video Design/` |
| Аудио-дизайн | `007 Проекты/003 Аудио дизайн Audio Designe & Production/` |
| 3D-моделирование | `007 Проекты/004 3D Моделирование 3D Modeling/` |
| Game-дизайн | `007 Проекты/005 Game Дизайн Game Design/` |
| Маркетинг | `007 Проекты/006 Маркетинг Marketing/` |
| Хобби вне рабочих направлений (без цели делать проект) | `008 Интересы (Увлечения)/` |

### По типу материала (если тема не совпадает ни с одним разделом выше)

| Тип материала | Куда |
|---|---|
| Сырое, не разобрано | `raw-data-repository/001 Входящие (Сырые данные)/` |
| Про себя, ценности, стиль мышления | `002 Мышление (Карта личности)/` |
| Конспект, модель, наблюдение | `003 Знания/` |
| Софт/промпт/скилл, которым реально пользуюсь | `004 Инструменты/` |
| Книга, курс, статья, референс (PDF, docx, ссылка, видео-курс) | `005 Ресурсы/` |
| Конкретное действие / цель | `006 Задачи (Цели)/` |
| Активный или будущий проект (не подходящий ни под одну тему выше) | `007 Проекты/` |
| Завершено / больше не актуально | `010 Архив (Неиспользуемое)/` |
| Настройки/архитектура самого проекта | `project-config/` |

Формат файла (PDF/docx/excel/ссылка/видео/аудио/голосовое/изображение и т.д.) определяет
только то, **как** материал обрабатывается технически при индексации (`kf/extract.py`:
текст читается напрямую, видео — транскрипция + разбор кадров, изображение — OCR/vision-caption
и т.д.) — не то, в какой раздел он попадает.

## Журнал знаний (Журнал знаний.md)

Автоматический лог каждого добавления/изменения/удаления файла в `raw-data-repository/`.
Пополняется сам при каждом запуске `kf.py ingest` (в т.ч. если файлы менялись вручную,
напрямую в Obsidian, без участия ассистента) — редактировать вручную не нужно и не имеет
смысла, правки перезапишутся при следующем `ingest`. При обнаружении удалённых файлов
запись попадает в журнал, но векторы/строки в Qdrant/Postgres/MinIO не удаляются
автоматически — это отдельное решение, принимается пользователем по запросу ассистента.
```

- [ ] **Step 2: Обновить `CLAUDE.md`**

Найти блок:

```markdown
## Регламент автодобавления знаний

Когда пользователь присылает новый материал (текст, голосовое, ссылку на видео/статью,
скриншот, PDF, репозиторий и т.п.) с пометкой "добавь в базу знаний":

1. Определи, к какому разделу `raw-data-repository/` это относится (Входящие, если не
   очевидно сразу — Знания, Ресурсы, Проекты, Задачи/Цели и т.д.).
2. Сохрани/законспектируй материал в этот раздел.
3. Сообщи пользователю, куда именно добавил (путь к файлу).
4. Обнови `карта_бз.md`, если появился новый файл или папка.
```

Заменить на:

```markdown
## Регламент автодобавления знаний

Когда пользователь присылает новый материал (текст, голосовое, ссылку на видео/статью,
скриншот, PDF, репозиторий, excel-таблицу, фото, изображение и т.п.) с пометкой "добавь в
базу знаний":

1. Определи, к какому разделу `raw-data-repository/` это относится — по таблице
   "Куда класть новое" в `карта_бз.md` (единственный источник правды для маршрутизации):
   сначала проверь совпадение по теме (Маркетинг/3D/Веб-дизайн/Видео/Аудио/Game/хобби —
   тема важнее формата файла), и только если тема не совпадает ни с одним существующим
   разделом — используй общее правило по типу материала.
2. Сохрани/законспектируй материал в этот раздел.
3. Запусти `kf.py ingest`, чтобы материал сразу попал в векторный поиск и в
   `Журнал знаний.md` (журнал пополняется автоматически при индексации, редактировать его
   вручную не нужно).
4. Сообщи пользователю, куда именно добавил (путь к файлу).
5. Обнови `карта_бз.md`, если появился новый файл или папка верхнего уровня.
```

- [ ] **Step 3: Обновить `AGENTS.md` тем же изменением, что и Step 2**

`AGENTS.md` содержит идентичный блок "Регламент автодобавления знаний" (дублирует
`CLAUDE.md` для Codex CLI) — применить точно ту же замену, что в Step 2.

- [ ] **Step 4: Упомянуть журнал в `knowledge-factory/README.md`**

Найти абзац:

```markdown
`kf.py ingest` теперь дополнительно пишет для каждого проиндексированного
файла осмысленную LLM-заметку в `Синтезированные данные (synthesized-notes)/`
(рядом с `data/`) и сразу делает её доступной для `search`/`ask`.
```

Заменить на:

```markdown
`kf.py ingest` теперь дополнительно пишет для каждого проиндексированного
файла осмысленную LLM-заметку в `Синтезированные данные (synthesized-notes)/`
(рядом с `data/`) и сразу делает её доступной для `search`/`ask`.

Каждый запуск `kf.py ingest` также пополняет `../Журнал знаний.md` (в корне проекта) —
короткую запись на каждое добавление/изменение/удаление файла в `raw-data-repository/`,
включая изменения, сделанные вручную в Obsidian. Удалённые файлы только логируются —
их векторы/записи в Qdrant/Postgres/MinIO не удаляются автоматически.
```

- [ ] **Step 5: Commit**

```bash
git add "карта_бз.md" CLAUDE.md AGENTS.md "knowledge-factory/README.md"
git commit -m "Digital brain | Регламент маршрутизации по теме + документация журнала знаний | V 1.4.31"
```
