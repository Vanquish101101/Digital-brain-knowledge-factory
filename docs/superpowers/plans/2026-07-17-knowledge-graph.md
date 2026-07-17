# Граф знаний (сущности, связи, Kuzu) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Автоматически извлекать сущности (люди/инструменты/проекты/темы/концепты) и связи
между ними из каждого проиндексированного файла, хранить в графовой БД Kuzu, и дать доступ
к прямым связям сущности через `kf.py graph <сущность>` и MCP-инструмент `graph_search`.

**Architecture:** Новый модуль `kf/graph.py` (промпт + разбор LLM-ответа, без сети/БД) +
новый модуль `kf/store/graph_store.py` (обёртка над Kuzu — встроенная графовая БД, файл на
диске, без Docker). Интеграция в существующий `kf/ingest.py` рядом с уже работающими
синтез-заметками — тот же текст файла, тот же паттерн деградации при ошибках. Доступ через
`kf.py graph` (CLI) и `graph_search` (MCP), по аналогии с уже существующими `search`/`ask`.

**Tech Stack:** Python 3.12, `kuzu` (новая зависимость, embedded graph DB), httpx (тот же
паттерн прямого HTTP-вызова OpenRouter, что и в `kf/synthesize.py`), pytest, click.

## Global Constraints

- Извлечение сущностей — автоматически на каждом `kf.py ingest`, тем же LLM-вызовом
  переиспользуя уже прочитанный текст файла (как синтез-заметки) — не отдельная команда.
- Сущность имеет тип: человек, инструмент, проект, тема, концепт (открытый список — LLM не
  жёстко ограничен этими значениями, но именно эти типы ожидаются как основные).
- Связь между двумя сущностями имеет: категорию из **строго фиксированного** списка
  (`использует`, `часть_проекта`, `связано_с_темой`, `автор_создатель`, `другое`),
  свободное текстовое описание, и путь к файлу-источнику.
- Дедупликация сущностей — по нормализованному имени (`strip()` + нижний регистр); одна и
  та же сущность из разных файлов должна быть одним узлом графа, не дубликатами.
- Сбой извлечения сущностей (сеть, некорректный JSON-ответ LLM) НЕ должен прерывать весь
  `ingest` — деградация с логированием, по тому же принципу, что уже применён к
  синтез-заметкам (`notes_failed`) и журналу знаний.
- Некорректный JSON-ответ LLM должен **поднимать `ValueError`**, а не молча возвращать
  пустые списки — иначе "сущностей в файле правда нет" неотличимо от "разбор не удался".
- Kuzu — файл на диске в `{DATA_ROOT}/graph/`, без нового `.env`-параметра (переиспользует
  уже существующий `DATA_ROOT`), без Docker-контейнера.
- Вне рамок этого плана: multi-hop запросы, визуализация графа, автоматическая очистка
  графа при удалении исходного файла, fuzzy-дедупликация сущностей (синонимы/опечатки).

---

### Task 1: Модуль `kf/graph.py` (извлечение — промпт и разбор ответа)

**Files:**
- Create: `knowledge-factory/kf/graph.py`
- Test: `knowledge-factory/tests/test_graph.py`

**Interfaces:**
- Consumes: ничего нового (только `kf.config.Settings` — уже существующий тип).
- Produces: `build_extraction_messages(text: str, source_path: str) -> list[dict]`,
  `parse_extraction_response(raw_response: str) -> tuple[list[dict], list[dict]]` (поднимает
  `ValueError` при некорректном JSON), `extract_entities_and_relationships(settings: Settings, text: str, source_path: str) -> tuple[list[dict], list[dict]]` —
  используются в Task 3 (`kf/ingest.py`). Каждая сущность в первом элементе кортежа — это
  `{"name": str, "type": str}`; каждая связь во втором — `{"from": str, "to": str, "category": str, "description": str}`.

- [ ] **Step 1: Write the failing tests**

Создать `knowledge-factory/tests/test_graph.py`:

```python
import json

import pytest

from kf.graph import build_extraction_messages, parse_extraction_response


def test_build_extraction_messages_includes_path_and_text():
    messages = build_extraction_messages("текст файла про Blender", "007 Проекты/файл.md")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "007 Проекты/файл.md" in messages[0]["content"]
    assert "текст файла про Blender" in messages[0]["content"]


def test_build_extraction_messages_lists_fixed_categories():
    messages = build_extraction_messages("текст", "файл.md")

    content = messages[0]["content"]
    for category in ["использует", "часть_проекта", "связано_с_темой", "автор_создатель", "другое"]:
        assert category in content


def test_parse_extraction_response_returns_entities_and_relationships():
    raw = json.dumps(
        {
            "entities": [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}],
            "relationships": [
                {"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг сцен"}
            ],
        }
    )

    entities, relationships = parse_extraction_response(raw)

    assert entities == [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}]
    assert relationships == [
        {"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг сцен"}
    ]


def test_parse_extraction_response_raises_on_invalid_json():
    with pytest.raises(ValueError):
        parse_extraction_response("это не json вообще")


def test_parse_extraction_response_raises_on_missing_top_level_keys():
    with pytest.raises(ValueError):
        parse_extraction_response(json.dumps({"entities": []}))


def test_parse_extraction_response_raises_on_entity_missing_fields():
    raw = json.dumps({"entities": [{"name": "Blender"}], "relationships": []})

    with pytest.raises(ValueError):
        parse_extraction_response(raw)


def test_parse_extraction_response_raises_on_relationship_missing_fields():
    raw = json.dumps(
        {"entities": [], "relationships": [{"from": "A", "to": "B", "category": "другое"}]}
    )

    with pytest.raises(ValueError):
        parse_extraction_response(raw)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_graph.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.graph'`.

- [ ] **Step 3: Implement**

Создать `knowledge-factory/kf/graph.py`:

```python
import json

import httpx

from kf.config import Settings

ENTITY_CATEGORIES = ["использует", "часть_проекта", "связано_с_темой", "автор_создатель", "другое"]

EXTRACTION_PROMPT_TEMPLATE = (
    "Ты строишь граф знаний личной базы пользователя. Ниже — текст, извлечённый из файла "
    "«{path}». Извлеки из текста сущности (люди, инструменты, проекты, темы, концепты) и "
    "связи между ними.\n\n"
    "Верни СТРОГО валидный JSON без пояснений и без markdown-разметки, в формате:\n"
    '{{"entities": [{{"name": "...", "type": "..."}}], '
    '"relationships": [{{"from": "...", "to": "...", "category": "...", "description": "..."}}]}}\n\n'
    "Поле category у каждой связи должно быть ровно одним из значений: "
    "использует, часть_проекта, связано_с_темой, автор_создатель, другое.\n\n"
    "Текст:\n{text}"
)


def build_extraction_messages(text: str, source_path: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(path=source_path, text=text)
    return [{"role": "user", "content": prompt}]


def parse_extraction_response(raw_response: str) -> tuple[list[dict], list[dict]]:
    try:
        data = json.loads(raw_response)
        entities = data["entities"]
        relationships = data["relationships"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"не удалось разобрать ответ извлечения сущностей: {exc}") from exc

    for entity in entities:
        if "name" not in entity or "type" not in entity:
            raise ValueError("сущность без обязательных полей name/type в ответе извлечения")
    for rel in relationships:
        if not all(key in rel for key in ("from", "to", "category", "description")):
            raise ValueError("связь без обязательных полей from/to/category/description в ответе извлечения")

    return entities, relationships


def extract_entities_and_relationships(
    settings: Settings, text: str, source_path: str
) -> tuple[list[dict], list[dict]]:
    messages = build_extraction_messages(text, source_path)
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.llm_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    return parse_extraction_response(raw_content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_graph.py -v`
Expected: PASS (7/7).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/graph.py" "knowledge-factory/tests/test_graph.py"
git commit -m "Digital brain | Модуль kf/graph.py: промпт и разбор ответа для извлечения сущностей | V 1.4.34"
```

---

### Task 2: Зависимость `kuzu` + модуль `kf/store/graph_store.py`

**Files:**
- Modify: `knowledge-factory/pyproject.toml` (добавить зависимость `kuzu`)
- Create: `knowledge-factory/kf/store/graph_store.py`
- Test: `knowledge-factory/tests/test_graph_store.py`

**Interfaces:**
- Consumes: `kf.config.Settings` (существующий тип, поле `data_root`).
- Produces: `normalize_entity_name(name: str) -> str`,
  `get_connection(settings: Settings) -> kuzu.Connection`, `ensure_schema(conn) -> None`,
  `upsert_entity(conn, name: str, entity_type: str) -> None`,
  `add_relationship(conn, from_name: str, to_name: str, category: str, description: str, source_path: str) -> None`,
  `query_entity(conn, name: str) -> list[dict]` (каждый элемент —
  `{"entity": str, "category": str, "description": str, "source_path": str}`). Используются
  в Task 3 (`kf/ingest.py`) и Task 4 (`kf/api.py`, `kf/cli.py`).

- [ ] **Step 1: Add the `kuzu` dependency**

Run: `cd "knowledge-factory" && uv add kuzu`

Проверить, что `knowledge-factory/pyproject.toml` теперь содержит `"kuzu>=..."` в
`dependencies`, и что `uv.lock` обновился.

- [ ] **Step 2: Write the failing tests**

Создать `knowledge-factory/tests/test_graph_store.py`:

```python
import pytest

from kf.config import Settings
from kf.store.graph_store import add_relationship, ensure_schema, get_connection, normalize_entity_name, query_entity, upsert_entity


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_graph_store.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.store.graph_store'`.

- [ ] **Step 4: Implement**

Создать `knowledge-factory/kf/store/graph_store.py`:

```python
from pathlib import Path

import kuzu

from kf.config import Settings


def normalize_entity_name(name: str) -> str:
    return name.strip().lower()


def get_connection(settings: Settings) -> kuzu.Connection:
    graph_dir = Path(settings.data_root) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(graph_dir))
    return kuzu.Connection(db)


def ensure_schema(conn: kuzu.Connection) -> None:
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Entity("
        "name STRING, display_name STRING, type STRING, PRIMARY KEY(name))"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS RELATED_TO("
        "FROM Entity TO Entity, category STRING, description STRING, source_path STRING)"
    )


def upsert_entity(conn: kuzu.Connection, name: str, entity_type: str) -> None:
    normalized = normalize_entity_name(name)
    result = conn.execute(
        "MATCH (e:Entity) WHERE e.name = $name RETURN e.name",
        parameters={"name": normalized},
    )
    if result.has_next():
        return
    conn.execute(
        "CREATE (e:Entity {name: $name, display_name: $display_name, type: $type})",
        parameters={"name": normalized, "display_name": name.strip(), "type": entity_type},
    )


def add_relationship(
    conn: kuzu.Connection,
    from_name: str,
    to_name: str,
    category: str,
    description: str,
    source_path: str,
) -> None:
    conn.execute(
        "MATCH (a:Entity), (b:Entity) WHERE a.name = $from_name AND b.name = $to_name "
        "CREATE (a)-[:RELATED_TO {category: $category, description: $description, source_path: $source_path}]->(b)",
        parameters={
            "from_name": normalize_entity_name(from_name),
            "to_name": normalize_entity_name(to_name),
            "category": category,
            "description": description,
            "source_path": source_path,
        },
    )


def query_entity(conn: kuzu.Connection, name: str) -> list[dict]:
    normalized = normalize_entity_name(name)
    result = conn.execute(
        "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) "
        "WHERE a.name = $name OR b.name = $name "
        "RETURN a.name, a.display_name, b.display_name, r.category, r.description, r.source_path",
        parameters={"name": normalized},
    )
    rows = []
    while result.has_next():
        a_name, a_display, b_display, category, description, source_path = result.get_next()
        other = b_display if a_name == normalized else a_display
        rows.append(
            {
                "entity": other,
                "category": category,
                "description": description,
                "source_path": source_path,
            }
        )
    return rows
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_graph_store.py -v`
Expected: PASS (5/5). Если сигнатура вызовов Kuzu (например, имя параметра `parameters=` или
метод `get_next()`) не совпадает с реально установленной версией пакета — ошибка будет
конкретной и понятной (например, `TypeError`); проверить фактическое API через
`uv run python -c "import kuzu; help(kuzu.Connection.execute)"` и поправить вызовы по месту,
сохранив сигнатуры функций `graph_store.py` неизменными (они зафиксированы в этой задаче и
на них полагаются задачи 3-4).

- [ ] **Step 6: Commit**

```bash
git add "knowledge-factory/pyproject.toml" "knowledge-factory/uv.lock" "knowledge-factory/kf/store/graph_store.py" "knowledge-factory/tests/test_graph_store.py"
git commit -m "Digital brain | Зависимость kuzu + kf/store/graph_store.py: хранение графа сущностей | V 1.4.35"
```

---

### Task 3: Интеграция в `kf/ingest.py` и `kf/api.py` (graph_conn, извлечение при ingest)

**Files:**
- Modify: `knowledge-factory/kf/api.py` (весь файл)
- Modify: `knowledge-factory/kf/ingest.py` (весь файл)
- Modify: `knowledge-factory/kf/cli.py:14-25` (`_build_ingest_deps`)
- Test: `knowledge-factory/tests/test_ingest.py`

**Interfaces:**
- Consumes: `extract_entities_and_relationships` из Task 1 (`kf.graph`); `get_connection`,
  `ensure_schema`, `upsert_entity`, `add_relationship` из Task 2 (`kf.store.graph_store`).
- Produces: `KnowledgeSession.graph_conn` (новое поле); `IngestDeps.graph_conn: object = None`
  (новое поле, по умолчанию `None` — существующие вызовы `IngestDeps(...)` без этого
  аргумента продолжают работать, граф-логика просто пропускается); `IngestStats` получает
  `entities_extracted: int = 0` и `entities_failed: int = 0`. Используются в Task 4.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_ingest.py`:

```python
def test_ingest_skips_entity_extraction_when_graph_conn_is_none(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.entities_extracted == 0
    assert stats.entities_failed == 0


def test_ingest_extracts_entities_when_graph_conn_provided(tmp_path, deps, monkeypatch):
    from kf.store.graph_store import ensure_schema, get_connection, query_entity

    graph_settings = deps.settings
    graph_settings.data_root = str(tmp_path.parent / "graph-data-extract")
    deps.graph_conn = get_connection(graph_settings)
    ensure_schema(deps.graph_conn)

    monkeypatch.setattr(
        "kf.ingest.extract_entities_and_relationships",
        lambda settings, text, source_path: (
            [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}],
            [{"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг"}],
        ),
    )
    (tmp_path / "note1.md").write_text("Заметка про Blender и Проект X.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.entities_extracted == 2
    results = query_entity(deps.graph_conn, "Blender")
    assert len(results) == 1
    assert results[0]["entity"] == "Проект X"


def test_entity_extraction_failure_does_not_abort_ingest(tmp_path, deps, monkeypatch):
    from kf.store.graph_store import ensure_schema, get_connection

    graph_settings = deps.settings
    graph_settings.data_root = str(tmp_path.parent / "graph-data-failure")
    deps.graph_conn = get_connection(graph_settings)
    ensure_schema(deps.graph_conn)

    def _boom(settings, text, source_path):
        raise RuntimeError("OpenRouter недоступен")

    monkeypatch.setattr("kf.ingest.extract_entities_and_relationships", _boom)
    (tmp_path / "note1.md").write_text("Заметка.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested == 2
    assert stats.entities_failed == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_ingest.py -k entity -v`
Expected: FAIL — `AttributeError: 'IngestDeps' object has no attribute 'graph_conn'` (и/или
`'IngestStats' object has no attribute 'entities_extracted'`).

- [ ] **Step 3: Implement — `kf/api.py`**

Заменить содержимое `knowledge-factory/kf/api.py` целиком на:

```python
from dataclasses import dataclass

from kf.config import Settings, load_settings
from kf.embeddings import embed, get_embedder
from kf.llm import build_prompt, call_llm
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.postgres import connect, ensure_schema
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import search as qdrant_search

COLLECTION = "knowledge"
VECTOR_SIZE = 384


@dataclass
class KnowledgeSession:
    settings: Settings
    pg_conn: object
    qdrant_client: object
    embedder: object
    graph_conn: object


def open_session() -> KnowledgeSession:
    settings = load_settings()

    pg_conn = connect(settings)
    ensure_schema(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    ensure_collection(qdrant_client, COLLECTION, vector_size=VECTOR_SIZE)

    embedder = get_embedder(settings)

    graph_conn = get_graph_connection(settings)
    ensure_graph_schema(graph_conn)

    return KnowledgeSession(
        settings=settings,
        pg_conn=pg_conn,
        qdrant_client=qdrant_client,
        embedder=embedder,
        graph_conn=graph_conn,
    )


def semantic_search(session: KnowledgeSession, query: str, limit: int = 5) -> list[dict]:
    vector = embed(session.embedder, [query])[0]
    results = qdrant_search(session.qdrant_client, COLLECTION, vector, limit=limit)
    return [
        {
            "path": r["payload"]["path"],
            "chunk_index": r["payload"]["chunk_index"],
            "text": r["payload"]["text"],
            "score": r["score"],
        }
        for r in results
    ]


def ask_question(session: KnowledgeSession, question: str, limit: int = 5) -> dict:
    results = semantic_search(session, question, limit=limit)
    messages = build_prompt(question, results)
    answer = call_llm(session.settings, messages)
    sources = sorted({r["path"] for r in results})
    return {"answer": answer, "sources": sources}


def get_stats(session: KnowledgeSession) -> dict:
    with session.pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]

    if session.qdrant_client.collection_exists(COLLECTION):
        chunk_count = session.qdrant_client.get_collection(COLLECTION).points_count
    else:
        chunk_count = 0

    return {"documents": doc_count, "chunks": chunk_count}
```

(Функция `graph_search` добавляется в Task 4 — здесь только плумбинг `graph_conn`.)

- [ ] **Step 4: Implement — `kf/ingest.py`**

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
from kf.graph import extract_entities_and_relationships
from kf.hashing import sha256_of_file
from kf.journal import append_entries, detect_deleted, extract_description, format_entry
from kf.scope import should_index
from kf.store.graph_store import add_relationship, upsert_entity
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
    graph_conn: object = None


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
    entities_extracted: int = 0
    entities_failed: int = 0


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


def _extract_and_store_entities(text: str, rel_key: str, deps: IngestDeps, stats: IngestStats) -> None:
    if deps.graph_conn is None:
        return
    try:
        entities, relationships = extract_entities_and_relationships(deps.settings, text, rel_key)
    except Exception as exc:
        print(f"[ingest] извлечение сущностей не удалось для {rel_key}: {exc}")
        stats.entities_failed += 1
        return

    for entity in entities:
        upsert_entity(deps.graph_conn, entity["name"], entity["type"])
    for rel in relationships:
        add_relationship(
            deps.graph_conn, rel["from"], rel["to"], rel["category"], rel["description"], rel_key
        )
    stats.entities_extracted += len(entities)


def ingest_directory(source_dir: Path, deps: IngestDeps, detect_deletions: bool = True) -> IngestStats:
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
            _extract_and_store_entities(text, rel_key, deps, stats)

        section = rel_key.split("/", 1)[0]
        description = extract_description(note_text) if note_text else ""
        action = "добавлено" if is_new else "изменено"
        journal_entries.append(format_entry(action, rel_key, section, description, today))

    if detect_deletions:
        notes_prefix = f"{notes_dir.name}/"
        try:
            known_paths = list_paths(deps.pg_conn, exclude_prefix=notes_prefix)
            deleted = detect_deleted(known_paths, seen_paths)
            for deleted_path in sorted(deleted):
                section = deleted_path.split("/", 1)[0]
                journal_entries.append(format_entry("удалено", deleted_path, section, "", today))
            stats.deleted_detected = len(deleted)
        except Exception as exc:
            print(f"[ingest] проверка удалённых файлов не удалась: {exc}")

    stats.journal_entries_written = len(journal_entries)
    try:
        append_entries(journal_entries, source_dir.parent / "Журнал знаний.md")
    except Exception as exc:
        print(f"[ingest] запись в Журнал знаний.md не удалась: {exc}")

    return stats
```

- [ ] **Step 5: Implement — `kf/cli.py` (`_build_ingest_deps`)**

В `knowledge-factory/kf/cli.py` найти:

```python
def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=COLLECTION,
        settings=settings,
    )
```

Заменить на:

```python
def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=COLLECTION,
        settings=settings,
        graph_conn=session.graph_conn,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_ingest.py tests/test_api.py tests/test_cli.py -v`
Expected: PASS — все тесты, включая три новых.

- [ ] **Step 7: Commit**

```bash
git add "knowledge-factory/kf/api.py" "knowledge-factory/kf/ingest.py" "knowledge-factory/kf/cli.py" "knowledge-factory/tests/test_ingest.py"
git commit -m "Digital brain | Извлечение сущностей интегрировано в kf.py ingest | V 1.4.36"
```

---

### Task 4: `graph_search` в `kf/api.py` + команда `kf.py graph`

**Files:**
- Modify: `knowledge-factory/kf/api.py` (добавить функцию в конец файла)
- Modify: `knowledge-factory/kf/cli.py` (добавить импорт и новую команду)
- Test: `knowledge-factory/tests/test_api.py`
- Test: `knowledge-factory/tests/test_cli.py`

**Interfaces:**
- Consumes: `query_entity` из Task 2 (`kf.store.graph_store`); `KnowledgeSession.graph_conn`
  из Task 3.
- Produces: `graph_search(session: KnowledgeSession, entity: str) -> list[dict]` в
  `kf/api.py` — используется в Task 5 (MCP-инструмент).

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_api.py`:

```python
from kf.api import KnowledgeSession, graph_search
from kf.config import load_settings
from kf.store.graph_store import add_relationship, ensure_schema, get_connection, upsert_entity


def test_graph_search_finds_direct_relationship(tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)
    upsert_entity(graph_conn, "Blender", "инструмент")
    upsert_entity(graph_conn, "Проект X", "проект")
    add_relationship(graph_conn, "Blender", "Проект X", "использует", "тестовая связь", "test.md")

    session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )

    results = graph_search(session, "Blender")

    assert len(results) == 1
    assert results[0]["entity"] == "Проект X"


def test_graph_search_returns_empty_for_unknown_entity(tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)

    session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )

    results = graph_search(session, "Несуществующая сущность")

    assert results == []
```

Добавить в `knowledge-factory/tests/test_cli.py`:

```python
def test_graph_command_reports_relationship(tmp_path, monkeypatch):
    from kf.api import KnowledgeSession
    from kf.store.graph_store import add_relationship, ensure_schema, get_connection, upsert_entity

    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)
    upsert_entity(graph_conn, "Blender", "инструмент")
    upsert_entity(graph_conn, "Проект X", "проект")
    add_relationship(graph_conn, "Blender", "Проект X", "использует", "рендеринг сцен", "note.md")

    fake_session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )
    monkeypatch.setattr("kf.cli.open_session", lambda: fake_session)

    runner = CliRunner()
    result = runner.invoke(cli, ["graph", "Blender"])

    assert result.exit_code == 0
    assert "Проект X" in result.output
    assert "использует" in result.output


def test_graph_command_reports_not_found_for_unknown_entity(tmp_path, monkeypatch):
    from kf.api import KnowledgeSession
    from kf.store.graph_store import ensure_schema, get_connection

    settings = load_settings()
    settings.data_root = str(tmp_path)
    graph_conn = get_connection(settings)
    ensure_schema(graph_conn)

    fake_session = KnowledgeSession(
        settings=settings, pg_conn=None, qdrant_client=None, embedder=None, graph_conn=graph_conn
    )
    monkeypatch.setattr("kf.cli.open_session", lambda: fake_session)

    runner = CliRunner()
    result = runner.invoke(cli, ["graph", "Несуществующая сущность"])

    assert result.exit_code == 0
    assert "не найдена" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_api.py tests/test_cli.py -k graph -v`
Expected: FAIL — `ImportError: cannot import name 'graph_search'` (и `Error: No such command 'graph'`).

- [ ] **Step 3: Implement — `kf/api.py`**

В `knowledge-factory/kf/api.py` найти существующий блок импортов (добавлен в Task 3):

```python
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
```

Заменить его на (добавлен третий импорт `query_entity`):

```python
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.graph_store import query_entity
```

Затем добавить в конец файла (после `get_stats`):

```python
def graph_search(session: KnowledgeSession, entity: str) -> list[dict]:
    return query_entity(session.graph_conn, entity)
```

- [ ] **Step 4: Implement — `kf/cli.py`**

Изменить строку импорта:

```python
from kf.api import COLLECTION, VECTOR_SIZE, ask_question, get_stats, open_session, semantic_search
```

на:

```python
from kf.api import COLLECTION, VECTOR_SIZE, ask_question, get_stats, graph_search, open_session, semantic_search
```

Добавить новую команду (после команды `search`, перед `ask` — порядок не важен, но для
консистентности расположить после `search`):

```python
@cli.command()
@click.argument("entity")
def graph(entity: str):
    """Показать все прямые связи сущности в графе знаний."""
    session = open_session()
    results = graph_search(session, entity)
    if not results:
        click.echo(f"Сущность «{entity}» не найдена в графе знаний.")
        return
    for r in results:
        click.echo(f"[{r['category']}] {r['entity']} — {r['description']} (из: {r['source_path']})")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_api.py tests/test_cli.py -v`
Expected: PASS — все тесты в обоих файлах.

- [ ] **Step 6: Commit**

```bash
git add "knowledge-factory/kf/api.py" "knowledge-factory/kf/cli.py" "knowledge-factory/tests/test_api.py" "knowledge-factory/tests/test_cli.py"
git commit -m "Digital brain | graph_search в kf/api.py + команда kf.py graph | V 1.4.37"
```

---

### Task 5: MCP-инструмент `graph_search`

**Files:**
- Modify: `knowledge-factory/kf/mcp_server.py` (весь файл)
- Test: `knowledge-factory/tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `graph_search` из Task 4 (`kf.api`).
- Produces: MCP-инструмент `graph_search(entity: str) -> list[dict]`, доступный через
  Knowledge MCP-сервер (используется агентами — мной и Hermes-ботом).

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_mcp_server.py`:

```python
@pytest.mark.anyio
async def test_lists_graph_search_tool():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.list_tools()
        names = {t.name for t in result.tools}

    assert "graph_search" in names


@pytest.mark.anyio
async def test_graph_search_tool_returns_empty_list_for_unknown_entity():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "graph_search", {"entity": "Несуществующая тестовая сущность xyz123"}
        )

    payload = json.loads(result.content[0].text)
    assert payload == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_mcp_server.py -k graph -v`
Expected: FAIL — `assert "graph_search" in names` не проходит (инструмента ещё нет).

- [ ] **Step 3: Implement**

Заменить содержимое `knowledge-factory/kf/mcp_server.py` целиком на:

```python
from mcp.server.fastmcp import FastMCP

from kf.api import KnowledgeSession
from kf.api import ask_question as _ask_question
from kf.api import get_stats as _get_stats
from kf.api import graph_search as _graph_search
from kf.api import open_session
from kf.api import semantic_search as _semantic_search

mcp = FastMCP("knowledge-factory")

_session: KnowledgeSession | None = None


def _get_session() -> KnowledgeSession:
    global _session
    if _session is None:
        _session = open_session()
    return _session


@mcp.tool()
def semantic_search(query: str, limit: int = 5) -> list[dict]:
    """Найти релевантные фрагменты в личной базе знаний (Digital Brain) по смыслу запроса."""
    return _semantic_search(_get_session(), query, limit=limit)


@mcp.tool()
def ask(question: str, limit: int = 5) -> dict:
    """Задать вопрос личной базе знаний и получить связный ответ со ссылками на источники."""
    return _ask_question(_get_session(), question, limit=limit)


@mcp.tool()
def stats() -> dict:
    """Сколько документов и чанков сейчас проиндексировано в базе знаний."""
    return _get_stats(_get_session())


@mcp.tool()
def graph_search(entity: str) -> list[dict]:
    """Найти прямые связи сущности в графе знаний (люди, инструменты, проекты, темы, концепты)."""
    return _graph_search(_get_session(), entity)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_mcp_server.py -v`
Expected: PASS — все тесты файла.

Затем прогнать полный набор тестов:

Run: `cd "knowledge-factory" && uv run python -m pytest -q`
Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/mcp_server.py" "knowledge-factory/tests/test_mcp_server.py"
git commit -m "Digital brain | MCP-инструмент graph_search | V 1.4.38"
```

---

### Task 6: Документация

**Files:**
- Modify: `knowledge-factory/README.md`
- Modify: `карта_бз.md`

Задача без тестов (документация) — TDD не применяется, отдельный шаг с коммитом.

- [ ] **Step 1: Обновить `knowledge-factory/README.md`**

Найти блок:

```markdown
## Knowledge MCP

`kf/mcp_server.py` (FastMCP) поверх `kf/api.py` — инструменты `semantic_search`, `ask`, `stats`.
```

Заменить на:

```markdown
## Knowledge MCP

`kf/mcp_server.py` (FastMCP) поверх `kf/api.py` — инструменты `semantic_search`, `ask`,
`stats`, `graph_search`.
```

Найти блок про синтез-заметки и журнал (после него добавить абзац про граф):

```markdown
Каждый запуск `kf.py ingest` также пополняет `../Журнал знаний.md` (в корне проекта) —
короткую запись на каждое добавление/изменение/удаление файла в `raw-data-repository/`,
включая изменения, сделанные вручную в Obsidian. Удалённые файлы только логируются —
их векторы/записи в Qdrant/Postgres/MinIO не удаляются автоматически.
```

Добавить сразу после этого абзаца:

```markdown
`kf.py ingest` также извлекает сущности (люди, инструменты, проекты, темы, концепты) и
связи между ними в граф знаний (Kuzu, `data/graph/`, без Docker). Посмотреть прямые связи
конкретной сущности: `uv run python kf.py graph "название сущности"`, или через MCP
(`graph_search`). Сбой извлечения не прерывает `ingest` — только логируется.
```

- [ ] **Step 2: Обновить `карта_бз.md`**

Найти в дереве структуры проекта строку с описанием `data/` внутри `knowledge-factory/`:

```
    └── data/                          — вектора/кэш (437MB+, растёт). Не в git (.gitignore).
        (qdrant/, minio/, redis/, graph/, model-cache/ — Postgres НЕ здесь, он на именованном
         томе Docker kf_postgres_data — Windows/NTFS не даёт контейнеру нужных Unix-прав)
```

(эта строка уже упоминает `graph/` как зарезервированный путь — оставить как есть, путь уже
верен и используется).

В самом конце файла, перед разделом "## История переименований и переносов", добавить новый
раздел:

```markdown
## Граф знаний (сущности и связи)

`kf.py ingest` автоматически извлекает сущности (люди/инструменты/проекты/темы/концепты) и
прямые связи между ними из каждого файла, хранит в Kuzu (`knowledge-factory/data/graph/`).
Посмотреть связи сущности: `kf.py graph "название"` или MCP-инструмент `graph_search`.
Multi-hop запросы и визуализация графа — не реализованы, запланированы как надстройка над
той же графовой схемой в будущем.
```

- [ ] **Step 3: Commit**

```bash
git add "knowledge-factory/README.md" "карта_бз.md"
git commit -m "Digital brain | Документация графа знаний | V 1.4.39"
```
