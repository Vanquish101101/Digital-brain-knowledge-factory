# Слой синтез-заметок для kf.py — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Научить `kf.py ingest` автоматически писать поверх обычных чанков ещё
и осмысленную LLM-заметку (о чём материал, ключевые идеи, применение) для
каждого успешно проиндексированного файла, сохранять её в отдельную папку
`Синтезированные данные (synthesized-notes)/` и сразу же делать её доступной
через `search`/`ask`.

**Architecture:** Новый модуль `kf/synthesize.py` (одна функция
`synthesize_note`, по образцу уже существующего `kf/vision_caption.py`).
`kf/ingest.py` дополняется: после успешной обычной индексации файла (без
изменений в этой части) вызывается синтез, результат пишется как `.md` и
доиндексируется тем же самым механизмом чанк→embed→store, что и всё
остальное. Ничего в существующем поведении (копия в MinIO, дедупликация по
хэшу) не меняется — синтез добавляется строго поверх, с собственной
изолированной обработкой ошибок, чтобы сбой синтеза никогда не терял уже
успешно записанные чанки исходного файла.

**Tech Stack:** Python 3.12, HTTP через `httpx` (уже используется в
`kf/llm.py`/`kf/vision_caption.py`), OpenRouter, тот же `LLM_MODEL`, что и для
`kf.py ask`. Тесты — `pytest`, monkeypatch для изоляции от реального
OpenRouter-вызова (по прецеденту `kf/vision_caption.py::caption_image`).

## Global Constraints

- Python ≥3.12, `pythonpath = ["."]` (см. `pyproject.toml`) — тесты запускаются
  из корня `Цифровой мозг (digital-brain)/`.
- Новая папка по умолчанию: `Синтезированные данные (synthesized-notes)/`,
  рядом с `data/` внутри `Цифровой мозг (digital-brain)/` — не внутри vault,
  не внутри `data/`.
- Синтез выполняется для **всех** типов файлов без исключений, встроен в
  `kf.py ingest` полностью автоматически (не отдельная команда).
- Копия сырого файла в MinIO и логика дедупликации по хэшу — без изменений,
  ничего не убирается из уже работающего пайплайна.
- Вызов `kf/synthesize.py::synthesize_note` **не покрывается автотестами** —
  по прецеденту `kf/vision_caption.py::caption_image`, сеть и деньги.
  Проверяется вручную живым прогоном. Автотестами покрывается только
  оркестрация в `kf/ingest.py` (с моком `synthesize_note`).
- Файлы, уже лежащие внутри `Синтезированные данные (synthesized-notes)/`, на
  повторный синтез не отправляются (защита от зацикливания — заметка про
  заметку).
- Каждая задача заканчивается зелёным прогоном `pytest` для затронутых файлов
  и коммитом в формате `Digital brain | что сделано | версия V X.Y.Z` (см.
  `CLAUDE.md`/историю коммитов проекта). План закоммичен как `V 1.4.18`,
  задачи этого плана продолжают нумерацию с `V 1.4.19`.

---

### Task 1: Настройка `SYNTHESIS_NOTES_DIR` в конфиге

**Files:**
- Modify: `kf/config.py` (весь файл)
- Modify: `tests/test_config.py`
- Modify: `tests/test_extract.py` (helper `_dummy_settings`)
- Modify: `.env.example`

**Interfaces:**
- Produces: `Settings.synthesis_notes_dir: str` — используется Task 2 (не
  напрямую) и Task 3 (`kf/ingest.py`).

- [ ] **Step 1: Написать падающий тест на дефолт новой настройки**

Добавить в конец `tests/test_config.py`:

```python
def test_defaults_synthesis_notes_dir(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("MINIO_ROOT_USER", "m")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "s")
    monkeypatch.delenv("SYNTHESIS_NOTES_DIR", raising=False)

    settings = load_settings()

    assert settings.synthesis_notes_dir == "./Синтезированные данные (synthesized-notes)"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_config.py::test_defaults_synthesis_notes_dir -v`
Expected: FAIL — `TypeError: Settings.__init__() got an unexpected keyword argument 'synthesis_notes_dir'` (поля ещё нет в тесте, вызывающем `load_settings()`, которая пока не проставляет его — реальная ошибка будет `AttributeError: 'Settings' object has no attribute 'synthesis_notes_dir'`).

- [ ] **Step 3: Добавить поле в `Settings` и `load_settings`**

Заменить содержимое `kf/config.py` целиком на:

```python
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    qdrant_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    data_root: str
    model_cache_dir: str
    embedding_model: str
    openrouter_api_key: str
    llm_model: str
    ocr_languages: str
    image_caption_threshold_chars: int
    vision_model: str
    video_frame_interval_seconds: int
    whisper_model_size: str
    max_video_frames: int
    synthesis_notes_dir: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        postgres_user=os.environ["POSTGRES_USER"],
        postgres_password=os.environ["POSTGRES_PASSWORD"],
        postgres_db=os.environ["POSTGRES_DB"],
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        minio_endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        minio_access_key=os.environ["MINIO_ROOT_USER"],
        minio_secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        data_root=os.environ.get("DATA_ROOT", "./data"),
        model_cache_dir=os.environ.get("MODEL_CACHE_DIR", "./data/model-cache"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash"),
        ocr_languages=os.environ.get("OCR_LANGUAGES", "rus+eng"),
        image_caption_threshold_chars=int(os.environ.get("IMAGE_CAPTION_THRESHOLD_CHARS", "20")),
        vision_model=os.environ.get("VISION_MODEL", "google/gemini-2.5-flash"),
        video_frame_interval_seconds=int(os.environ.get("VIDEO_FRAME_INTERVAL_SECONDS", "15")),
        whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "small"),
        max_video_frames=int(os.environ.get("MAX_VIDEO_FRAMES", "20")),
        synthesis_notes_dir=os.environ.get(
            "SYNTHESIS_NOTES_DIR", "./Синтезированные данные (synthesized-notes)"
        ),
    )
```

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_config.py -v`
Expected: PASS, все тесты в файле зелёные.

- [ ] **Step 5: Обновить `_dummy_settings` в `tests/test_extract.py`, иначе он сломается**

`Settings` теперь требует `synthesis_notes_dir` при прямом конструировании.
В `tests/test_extract.py` найти:

```python
        image_caption_threshold_chars=20, vision_model="v",
        video_frame_interval_seconds=15, whisper_model_size="small",
        max_video_frames=20,
    )
```

Заменить на:

```python
        image_caption_threshold_chars=20, vision_model="v",
        video_frame_interval_seconds=15, whisper_model_size="small",
        max_video_frames=20, synthesis_notes_dir="./notes",
    )
```

- [ ] **Step 6: Прогнать весь файл `test_extract.py`, убедиться что не сломался**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_extract.py -v`
Expected: PASS, все тесты зелёные (никакой тест не касается `synthesis_notes_dir`
по смыслу, только конструктор `Settings` не должен падать).

- [ ] **Step 7: Задокументировать переменную в `.env.example`**

В `.env.example` после строки `MAX_VIDEO_FRAMES=20` (или в конце файла, если
структура успела измениться — искать блок `# --- OCR / vision / видео ---`)
добавить:

```
# --- Синтез-заметки (осмысление контента поверх сырых чанков) ---
# Папка для LLM-заметок (о чём материал, ключевые идеи, применение),
# создаётся автоматически при первом ingest. Модель — тот же LLM_MODEL, что и для ask.
SYNTHESIS_NOTES_DIR=./Синтезированные данные (synthesized-notes)
```

- [ ] **Step 8: Commit**

```bash
cd "E:\Digital brain"
git add "Цифровой мозг (digital-brain)/kf/config.py" "Цифровой мозг (digital-brain)/tests/test_config.py" "Цифровой мозг (digital-brain)/tests/test_extract.py" "Цифровой мозг (digital-brain)/.env.example"
git commit -m "Digital brain | Настройка SYNTHESIS_NOTES_DIR в kf/config.py | V 1.4.19"
```

---

### Task 2: Модуль `kf/synthesize.py`

**Files:**
- Create: `Цифровой мозг (digital-brain)/kf/synthesize.py`
- Test: `Цифровой мозг (digital-brain)/tests/test_synthesize.py`

**Interfaces:**
- Consumes: `Settings` (из `kf/config.py`, Task 1) — поля `openrouter_api_key`, `llm_model`.
- Produces: `build_synthesis_messages(text: str, source_path: str) -> list[dict]`
  (чистая функция, тестируется), `synthesize_note(settings: Settings, text: str, source_path: str) -> str`
  (реальный HTTP-вызов, используется Task 3, автотестом не покрывается).

- [ ] **Step 1: Написать падающий тест на `build_synthesis_messages`**

Создать `tests/test_synthesize.py`:

```python
from kf.synthesize import build_synthesis_messages


def test_build_synthesis_messages_includes_path_and_text():
    messages = build_synthesis_messages("некий текст файла", "путь/файл.md")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "путь/файл.md" in messages[0]["content"]
    assert "некий текст файла" in messages[0]["content"]


def test_build_synthesis_messages_asks_for_three_part_structure():
    messages = build_synthesis_messages("текст", "файл.md")

    content = messages[0]["content"]
    assert "ключевые идеи" in content.lower()
    assert "пригодиться" in content.lower()
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_synthesize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kf.synthesize'`.

- [ ] **Step 3: Написать `kf/synthesize.py`**

Создать `kf/synthesize.py`:

```python
import httpx

from kf.config import Settings

SYNTHESIS_PROMPT_TEMPLATE = (
    "Ты помогаешь строить личную базу знаний пользователя. Ниже — текст, "
    "извлечённый из файла «{path}». Напиши структурированную заметку на "
    "русском языке из трёх частей:\n"
    "1. О чём этот материал (кратко, 1-2 предложения).\n"
    "2. Ключевые идеи и факты.\n"
    "3. Как это может пригодиться в будущих задачах и проектах.\n\n"
    "Текст:\n{text}"
)


def build_synthesis_messages(text: str, source_path: str) -> list[dict]:
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(path=source_path, text=text)
    return [{"role": "user", "content": prompt}]


def synthesize_note(settings: Settings, text: str, source_path: str) -> str:
    messages = build_synthesis_messages(text, source_path)
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.llm_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_synthesize.py -v`
Expected: PASS, 2/2.

- [ ] **Step 5: Commit**

```bash
cd "E:\Digital brain"
git add "Цифровой мозг (digital-brain)/kf/synthesize.py" "Цифровой мозг (digital-brain)/tests/test_synthesize.py"
git commit -m "Digital brain | Модуль синтез-заметок kf/synthesize.py через OpenRouter | V 1.4.20"
```

---

### Task 3: Оркестрация синтеза в `kf/ingest.py`

**Files:**
- Modify: `Цифровой мозг (digital-brain)/kf/ingest.py` (весь файл)
- Modify: `Цифровой мозг (digital-brain)/tests/test_ingest.py` (весь файл)

**Interfaces:**
- Consumes: `synthesize_note(settings, text, source_path) -> str` (из Task 2, импортируется как `kf.synthesize.synthesize_note` — именно так его нужно мокать в тестах через `monkeypatch.setattr("kf.ingest.synthesize_note", ...)`, т.к. импортируется в `kf/ingest.py` по имени).
- Produces: `IngestStats.notes_synthesized: int`, `IngestStats.notes_failed: int` — используются Task 4 (`kf/cli.py`).

Этот таск — основной объём работы. Он трогает уже существующий,
интеграционно протестированный (без моков, реальный Postgres/Qdrant/MinIO)
`ingest_directory`. Каждый существующий тест в `tests/test_ingest.py`
пересматривается: часть перестаёт совпадать по числам (появляется
дополнительный файл-заметка на каждый успешно проиндексированный файл), часть
остаётся без изменений. Ниже — полный обновлённый файл теста и полный
обновлённый `kf/ingest.py`, никаких сокращений.

- [ ] **Step 1: Обновить `kf/ingest.py` — падающая часть придёт из теста на Step 3**

Заменить содержимое `kf/ingest.py` целиком на:

```python
import uuid
from dataclasses import dataclass
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embeddings import embed
from kf.extract import extract_text
from kf.hashing import sha256_of_file
from kf.scope import should_index
from kf.store.minio_store import upload_file
from kf.store.postgres import needs_ingest, record_ingested
from kf.store.qdrant_store import upsert_chunks
from kf.synthesize import synthesize_note

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")
NOTES_PATH_PREFIX = "Синтезированные данные (synthesized-notes)"


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
) -> None:
    try:
        note_text = synthesize_note(deps.settings, text, rel_key)
    except Exception as exc:
        print(f"[ingest] синтез не удался для {rel_key}: {exc}")
        stats.notes_failed += 1
        return

    notes_dir = Path(deps.settings.synthesis_notes_dir)
    note_path = notes_dir / f"{rel_key}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_text, encoding="utf-8")
    stats.notes_synthesized += 1

    note_rel_key = f"{NOTES_PATH_PREFIX}/{rel_key}.md"
    note_hash = sha256_of_file(note_path)
    if needs_ingest(deps.pg_conn, note_rel_key, note_hash):
        _store_text(note_text, note_rel_key, note_path, deps, stats)
        record_ingested(deps.pg_conn, note_rel_key, note_hash)
        stats.files_ingested += 1


def ingest_directory(source_dir: Path, deps: IngestDeps) -> IngestStats:
    stats = IngestStats()
    notes_dir = Path(deps.settings.synthesis_notes_dir).resolve()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or not should_index(path):
            continue

        rel_key = path.relative_to(source_dir).as_posix()
        stats.files_scanned += 1

        file_hash = sha256_of_file(path)
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
        if not is_note:
            _synthesize_and_index_note(text, rel_key, deps, stats)

    return stats
```

- [ ] **Step 2: Проверить импорт не ломает остальной пакет**

Run: `cd "Цифровой мозг (digital-brain)" && uv run python -c "import kf.ingest"`
Expected: без ошибок (модуль импортируется, `kf.synthesize` уже существует
после Task 2).

- [ ] **Step 3: Переписать `tests/test_ingest.py` целиком**

Заменить содержимое `tests/test_ingest.py` целиком на:

```python
from pathlib import Path

import pytest

from kf.config import load_settings
from kf.embeddings import get_embedder
from kf.ingest import IngestDeps, ingest_directory
from kf.store.minio_store import ensure_bucket, file_exists, get_client as get_minio_client
from kf.store.postgres import connect, ensure_schema
from kf.store.qdrant_store import ensure_collection, get_client as get_qdrant_client

COLLECTION = "kf_test_ingest"


@pytest.fixture
def deps(tmp_path_factory, monkeypatch):
    settings = load_settings()
    settings.synthesis_notes_dir = str(tmp_path_factory.mktemp("synth-notes"))
    monkeypatch.setattr(
        "kf.ingest.synthesize_note",
        lambda settings, text, source_path: f"Синтез-заметка про {source_path}",
    )

    pg_conn = connect(settings)
    ensure_schema(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    ensure_collection(qdrant_client, COLLECTION, vector_size=384)

    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)

    embedder = get_embedder(settings)

    d = IngestDeps(
        pg_conn=pg_conn,
        qdrant_client=qdrant_client,
        minio_client=minio_client,
        embedder=embedder,
        collection=COLLECTION,
        settings=settings,
    )
    yield d

    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE '%note%'")
    pg_conn.commit()
    qdrant_client.delete_collection(COLLECTION)
    pg_conn.close()


def test_ingests_new_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка про борщ и его рецепт.", encoding="utf-8")
    (tmp_path / "note2.md").write_text("Заметка про Docker и контейнеры.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 2
    assert stats.files_ingested == 4
    assert stats.files_skipped == 0
    assert stats.chunks_written == 4
    assert stats.notes_synthesized == 2
    assert file_exists(deps.minio_client, "note1.md") is True
    assert file_exists(deps.minio_client, "note2.md") is True


def test_second_run_skips_unchanged_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка про борщ и его рецепт.", encoding="utf-8")

    ingest_directory(tmp_path, deps)
    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 0
    assert stats.files_skipped == 1


def test_changed_file_gets_reingested(tmp_path, deps):
    f = tmp_path / "note1.md"
    f.write_text("Старый текст.", encoding="utf-8")
    ingest_directory(tmp_path, deps)

    f.write_text("Новый текст после правки.", encoding="utf-8")
    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested == 1
    assert stats.files_skipped == 0


def test_ignores_excluded_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")
    (tmp_path / "archive.zip").write_bytes(b"not a real archive")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 2
    assert stats.notes_synthesized == 1


def test_nested_file_uses_forward_slash_object_key(tmp_path, deps):
    subdir = tmp_path / "001 Подпапка (со скобками)"
    subdir.mkdir()
    (subdir / "note1.md").write_text("Вложенная заметка.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    assert file_exists(deps.minio_client, "001 Подпапка (со скобками)/note1.md") is True


def test_failed_file_does_not_abort_whole_run(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")
    (tmp_path / "broken.png").write_bytes(b"not a real png")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 2
    assert stats.files_ingested == 2
    assert stats.files_failed == 1


def test_writes_synthesized_note_file_to_notes_dir(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    note_path = Path(deps.settings.synthesis_notes_dir) / "note1.md.md"
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "Синтез-заметка про note1.md"


def test_synthesis_failure_does_not_lose_source_chunks(tmp_path, deps, monkeypatch):
    (tmp_path / "note1.md").write_text("Годная заметка, но синтез упадёт.", encoding="utf-8")

    def _boom(settings, text, source_path):
        raise RuntimeError("OpenRouter недоступен")

    monkeypatch.setattr("kf.ingest.synthesize_note", _boom)

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 1
    assert stats.chunks_written == 1
    assert stats.notes_failed == 1
    assert stats.notes_synthesized == 0


def test_files_inside_notes_dir_are_not_resynthesized(tmp_path, deps):
    notes_dir = Path(deps.settings.synthesis_notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "already-a-note.md").write_text("Уже готовая синтез-заметка.", encoding="utf-8")

    stats = ingest_directory(notes_dir, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 1
    assert stats.notes_synthesized == 0
```

- [ ] **Step 4: Прогнать полный файл теста**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_ingest.py -v`
Expected: PASS, 9/9. Если что-то падает на конкретных числах (`files_ingested`,
`chunks_written`) — перепроверить, что `_synthesize_and_index_note` реально
инкрементирует `stats.files_ingested` при успешном сохранении заметки (Step 1),
а не только `notes_synthesized`.

- [ ] **Step 5: Прогнать весь пакет тестов, убедиться в отсутствии регрессий**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest -q`
Expected: все тесты, кроме `tests/test_ingest.py`/`tests/test_cli.py`
(которые ещё не запускались с новым кодом до этого шага в изоляции), должны
быть не затронуты. `test_cli.py::test_ingest_reports_summary` в этот момент,
скорее всего, упадёт или сделает реальный сетевой вызов — это ожидаемо,
чинится в Task 4.

- [ ] **Step 6: Commit**

```bash
cd "E:\Digital brain"
git add "Цифровой мозг (digital-brain)/kf/ingest.py" "Цифровой мозг (digital-brain)/tests/test_ingest.py"
git commit -m "Digital brain | Оркестрация синтез-заметок в kf/ingest.py | V 1.4.21"
```

---

### Task 4: Вывод статистики в CLI + документация

**Files:**
- Modify: `Цифровой мозг (digital-brain)/kf/cli.py`
- Modify: `Цифровой мозг (digital-brain)/tests/test_cli.py`
- Modify: `Цифровой мозг (digital-brain)/README.md`

**Interfaces:**
- Consumes: `IngestStats.notes_synthesized`, `IngestStats.notes_failed` (из Task 3).

- [ ] **Step 1: Написать падающий тест на новую строку в выводе `ingest`**

Заменить в `tests/test_cli.py` функцию `test_ingest_reports_summary` на:

```python
def test_ingest_reports_summary(tmp_path, monkeypatch):
    (tmp_path / "cli-note.md").write_text("Заметка для теста CLI ingest.", encoding="utf-8")
    monkeypatch.setenv("SYNTHESIS_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setattr(
        "kf.ingest.synthesize_note",
        lambda settings, text, source_path: f"Заметка про {source_path}",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["ingest", "--source", str(tmp_path)])

    assert result.exit_code == 0
    assert "проиндексировано: 2" in result.output
    assert "заметок синтезировано: 1" in result.output

    settings = load_settings()
    conn = connect(settings)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE path LIKE '%cli-note%'")
    conn.commit()
    conn.close()
```

(Импорты `from click.testing import CliRunner`, `from kf.cli import cli`,
`from kf.config import load_settings`, `from kf.store.postgres import connect`
в начале файла уже есть, менять не нужно.)

- [ ] **Step 2: Убедиться, что тест падает**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_cli.py::test_ingest_reports_summary -v`
Expected: FAIL — в выводе нет строки `"заметок синтезировано:"`, т.к.
`kf/cli.py` её ещё не печатает.

- [ ] **Step 3: Обновить вывод в `kf/cli.py`**

В `kf/cli.py` найти:

```python
    stats = ingest_directory(src, deps)
    click.echo(
        f"Готово. просканировано: {stats.files_scanned}, "
        f"проиндексировано: {stats.files_ingested}, "
        f"пропущено (без изменений): {stats.files_skipped}, "
        f"ошибок: {stats.files_failed}, "
        f"чанков записано: {stats.chunks_written}"
    )
```

Заменить на:

```python
    stats = ingest_directory(src, deps)
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

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_cli.py -v`
Expected: PASS, все тесты в файле зелёные.

- [ ] **Step 5: Прогнать весь набор тестов**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest -q`
Expected: все тесты зелёные (полный счётчик вырастет на количество новых
тестов из Task 2 и Task 3 по сравнению с базовым состоянием перед этим
планом).

- [ ] **Step 6: Обновить `README.md`**

В `README.md` (`Цифровой мозг (digital-brain)/README.md`) найти строку:

```
- `kf.py` (ingest/search/ask/stats) — готов, 51 тест (TDD).
```

Заменить на актуальное число тестов, полученное на Step 5 (запустить
`uv run pytest -q` и посчитать по итоговой строке `N passed`), например:

```
- `kf.py` (ingest/search/ask/stats) — готов, N тест (TDD), включая
  автоматический слой синтез-заметок поверх обычной индексации.
```

Там же, в разделе `## Команды kf.py`, после блока с командами добавить:

```
`kf.py ingest` теперь дополнительно пишет для каждого проиндексированного
файла осмысленную LLM-заметку в `Синтезированные данные (synthesized-notes)/`
(рядом с `data/`) и сразу делает её доступной для `search`/`ask`.
```

- [ ] **Step 7: Commit**

```bash
cd "E:\Digital brain"
git add "Цифровой мозг (digital-brain)/kf/cli.py" "Цифровой мозг (digital-brain)/tests/test_cli.py" "Цифровой мозг (digital-brain)/README.md"
git commit -m "Digital brain | Статистика синтез-заметок в kf.py ingest + README | V 1.4.22"
```

---

## После выполнения плана (не отдельная задача, ручной шаг)

- Ручная сквозная проверка на реальных файлах (как делалось для
  картинок/видео): взять пару реальных заметок/PDF/картинок/видео, прогнать
  `kf.py ingest`, убедиться что в
  `Синтезированные данные (synthesized-notes)/` появились осмысленные `.md`,
  и что `kf.py ask` находит их наравне с сырыми чанками.
- Обновить `Файлы настройки проекта/РЕШЕНИЯ-И-СТАТУС.md` и `карта_бз.md` по
  итогам (по аналогии с тем, как это делалось после фичи картинок/видео) —
  вне рамок этого implementation-плана, отдельный шаг после того, как весь
  план выполнен и проверен.
