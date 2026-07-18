# Переключаемые модели эмбеддинга — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Дать возможность быстро переключаться между несколькими моделями эмбеддинга
(локальная MiniLM, Qwen3-Embedding-8B и OpenAI text-embedding-3-small/large через
OpenRouter) одной CLI-командой, без потери уже посчитанных векторов и без случайной траты
API-бюджета.

**Architecture:** Фиксированный реестр профилей моделей (`kf/embedding_models.py`) + файл
состояния активного профиля на диске (`kf/embedding_state.py`) + новый провайдер эмбеддинга
через OpenRouter (`kf/embeddings.py`, тем же паттерном прямого HTTP-вызова, что уже есть в
`kf/graph.py`). У каждого профиля своя коллекция в Qdrant — переключение на уже
использовавшийся профиль мгновенное. `kf.py search`/`ask`/`ingest` читают активный профиль
динамически. Отдельная команда `kf.py embedding-model sync` досчитывает эмбеддинги только для
файлов, которых не хватает в коллекции активного профиля — не трогает LLM-синтез заметок и
граф знаний.

**Tech Stack:** Python 3.12, httpx (уже используется для OpenRouter в `kf/graph.py`), Qdrant
(`qdrant-client`, уже используется), pytest, click.

## Global Constraints

- Профиль `local` обязан использовать существующее имя коллекции `knowledge` и размерность
  384 — без миграции уже накопленных данных.
- Переключение активного профиля (`embedding-model use`) НЕ должно само по себе вызывать
  платные API-запросы — это только смена состояния плюс бесплатная сверка покрытия
  (сравнение множеств путей, без сети).
- Только `kf.py embedding-model sync` тратит API-бюджет (если активный профиль не `local`), и
  только для реально отсутствующих в коллекции файлов — не переделывает LLM-синтез заметок и
  извлечение сущностей графа.
- Список моделей — фиксированный словарь в коде (`kf/embedding_models.py`), не внешний
  конфиг-файл — соответствует стилю всего проекта (`.env` + код).
- Сбой сетевого вызова к OpenRouter при эмбеддинге одного файла не должен прерывать всю
  операцию `sync`/`ingest` — лог и пропуск, по тому же принципу, что уже применён к
  извлечению сущностей графа в `kf/ingest.py`.
- Вне рамок этого плана: автоматический реиндекс при переключении, UI/кнопки (CLI и есть
  интерфейс), изменение графа знаний или LLM-синтеза заметок, произвольный конфиг моделей.

---

### Task 1: Реестр моделей — `kf/embedding_models.py`

**Files:**
- Create: `knowledge-factory/kf/embedding_models.py`
- Test: `knowledge-factory/tests/test_embedding_models.py`

**Interfaces:**
- Consumes: ничего нового.
- Produces: `EmbeddingProfile` (dataclass: `name: str`, `provider: str`, `model_id: str`,
  `dimension: int`, `collection: str`), `EMBEDDING_PROFILES: dict[str, EmbeddingProfile]`,
  `DEFAULT_PROFILE_NAME: str`, `get_profile(name: str) -> EmbeddingProfile` (поднимает
  `ValueError` для неизвестного имени). Используются во всех последующих задачах.

- [ ] **Step 1: Write the failing tests**

Создать `knowledge-factory/tests/test_embedding_models.py`:

```python
import pytest

from kf.embedding_models import DEFAULT_PROFILE_NAME, EMBEDDING_PROFILES, get_profile


def test_default_profile_is_local_and_matches_existing_collection():
    assert DEFAULT_PROFILE_NAME == "local"
    local = EMBEDDING_PROFILES["local"]
    assert local.provider == "local"
    assert local.collection == "knowledge"
    assert local.dimension == 384


def test_all_expected_profiles_are_registered():
    assert set(EMBEDDING_PROFILES.keys()) == {"local", "qwen3-8b", "openai-small", "openai-large"}


def test_openrouter_profiles_have_openrouter_provider():
    for name in ("qwen3-8b", "openai-small", "openai-large"):
        assert EMBEDDING_PROFILES[name].provider == "openrouter"


def test_collection_names_are_unique():
    collections = [p.collection for p in EMBEDDING_PROFILES.values()]
    assert len(collections) == len(set(collections))


def test_get_profile_returns_matching_profile():
    profile = get_profile("openai-small")
    assert profile.model_id == "openai/text-embedding-3-small"
    assert profile.dimension == 1536


def test_get_profile_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_profile("не-существует")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embedding_models.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.embedding_models'`.

- [ ] **Step 3: Implement**

Создать `knowledge-factory/kf/embedding_models.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    provider: str
    model_id: str
    dimension: int
    collection: str


DEFAULT_PROFILE_NAME = "local"

EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "local": EmbeddingProfile(
        name="local",
        provider="local",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension=384,
        collection="knowledge",
    ),
    "qwen3-8b": EmbeddingProfile(
        name="qwen3-8b",
        provider="openrouter",
        model_id="qwen/qwen3-embedding-8b",
        dimension=4096,
        collection="knowledge__qwen3_8b",
    ),
    "openai-small": EmbeddingProfile(
        name="openai-small",
        provider="openrouter",
        model_id="openai/text-embedding-3-small",
        dimension=1536,
        collection="knowledge__openai_small",
    ),
    "openai-large": EmbeddingProfile(
        name="openai-large",
        provider="openrouter",
        model_id="openai/text-embedding-3-large",
        dimension=3072,
        collection="knowledge__openai_large",
    ),
}


def get_profile(name: str) -> EmbeddingProfile:
    if name not in EMBEDDING_PROFILES:
        raise ValueError(f"неизвестный профиль эмбеддинга: {name}")
    return EMBEDDING_PROFILES[name]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embedding_models.py -v`
Expected: PASS (6/6).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/embedding_models.py" "knowledge-factory/tests/test_embedding_models.py"
git commit -m "Digital brain | Реестр профилей эмбеддинга: kf/embedding_models.py | V 1.4.41"
```

---

### Task 2: Состояние активного профиля — `kf/embedding_state.py`

**Files:**
- Create: `knowledge-factory/kf/embedding_state.py`
- Test: `knowledge-factory/tests/test_embedding_state.py`

**Interfaces:**
- Consumes: `DEFAULT_PROFILE_NAME` из Task 1 (`kf.embedding_models`).
- Produces: `get_active_profile_name(data_root: str) -> str`,
  `set_active_profile_name(data_root: str, name: str) -> None`. Используются в Task 5 и Task 7.

- [ ] **Step 1: Write the failing tests**

Создать `knowledge-factory/tests/test_embedding_state.py`:

```python
from kf.embedding_state import get_active_profile_name, set_active_profile_name


def test_returns_default_when_no_state_file(tmp_path):
    assert get_active_profile_name(str(tmp_path)) == "local"


def test_set_then_get_roundtrip(tmp_path):
    set_active_profile_name(str(tmp_path), "qwen3-8b")

    assert get_active_profile_name(str(tmp_path)) == "qwen3-8b"


def test_set_overwrites_previous_value(tmp_path):
    set_active_profile_name(str(tmp_path), "qwen3-8b")
    set_active_profile_name(str(tmp_path), "openai-large")

    assert get_active_profile_name(str(tmp_path)) == "openai-large"


def test_get_strips_whitespace_from_state_file(tmp_path):
    state_dir = tmp_path
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active_embedding_model.txt").write_text("  openai-small  \n", encoding="utf-8")

    assert get_active_profile_name(str(tmp_path)) == "openai-small"


def test_set_creates_data_root_if_missing(tmp_path):
    nested = tmp_path / "nested" / "data"

    set_active_profile_name(str(nested), "local")

    assert get_active_profile_name(str(nested)) == "local"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embedding_state.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.embedding_state'`.

- [ ] **Step 3: Implement**

Создать `knowledge-factory/kf/embedding_state.py`:

```python
from pathlib import Path

from kf.embedding_models import DEFAULT_PROFILE_NAME

STATE_FILENAME = "active_embedding_model.txt"


def _state_file(data_root: str) -> Path:
    return Path(data_root) / STATE_FILENAME


def get_active_profile_name(data_root: str) -> str:
    path = _state_file(data_root)
    if not path.exists():
        return DEFAULT_PROFILE_NAME
    name = path.read_text(encoding="utf-8").strip()
    return name or DEFAULT_PROFILE_NAME


def set_active_profile_name(data_root: str, name: str) -> None:
    path = _state_file(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embedding_state.py -v`
Expected: PASS (5/5).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/embedding_state.py" "knowledge-factory/tests/test_embedding_state.py"
git commit -m "Digital brain | Состояние активного профиля эмбеддинга: kf/embedding_state.py | V 1.4.42"
```

---

### Task 3: Провайдер эмбеддинга через OpenRouter — `kf/embeddings.py`

**Files:**
- Modify: `knowledge-factory/kf/embeddings.py` (только добавление новых функций в конец файла
  — существующие `get_embedder`/`embed` не трогать)
- Test: `knowledge-factory/tests/test_embeddings.py` (только добавление новых тестов —
  существующие два теста не трогать)

**Interfaces:**
- Consumes: `EmbeddingProfile` из Task 1 (`kf.embedding_models`).
- Produces: `build_embedding_request(profile, texts) -> dict`,
  `parse_embedding_response(raw_json: dict) -> list[list[float]]`,
  `embed_via_openrouter(settings, profile, texts) -> list[list[float]]`,
  `get_embedder_for_profile(settings, profile) -> object | None` (возвращает `None` для
  провайдера `openrouter` — там нет постоянного объекта-клиента),
  `embed_for_profile(settings, profile, embedder, texts) -> list[list[float]]`. Используются
  в Task 5, 6, 7.

- [ ] **Step 1: Write the failing tests**

Добавить в конец `knowledge-factory/tests/test_embeddings.py`:

```python
from kf.embedding_models import EMBEDDING_PROFILES
from kf.embeddings import build_embedding_request, embed_for_profile, get_embedder_for_profile, parse_embedding_response


def test_build_embedding_request_includes_model_and_texts():
    profile = EMBEDDING_PROFILES["openai-small"]

    request = build_embedding_request(profile, ["привет", "мир"])

    assert request == {"model": "openai/text-embedding-3-small", "input": ["привет", "мир"]}


def test_parse_embedding_response_extracts_vectors_in_order():
    raw = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

    vectors = parse_embedding_response(raw)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_get_embedder_for_profile_returns_none_for_openrouter_provider():
    settings = load_settings()
    profile = EMBEDDING_PROFILES["openai-small"]

    embedder = get_embedder_for_profile(settings, profile)

    assert embedder is None


def test_get_embedder_for_profile_returns_local_embedder_for_local_provider():
    settings = load_settings()
    profile = EMBEDDING_PROFILES["local"]

    embedder = get_embedder_for_profile(settings, profile)
    vectors = embed_for_profile(settings, profile, embedder, ["текст"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embeddings.py -v`
Expected: FAIL с `ImportError: cannot import name 'build_embedding_request'` — два исходных
теста продолжают проходить.

- [ ] **Step 3: Implement**

Добавить в конец `knowledge-factory/kf/embeddings.py`:

```python
import httpx

from kf.embedding_models import EmbeddingProfile


def build_embedding_request(profile: EmbeddingProfile, texts: list[str]) -> dict:
    return {"model": profile.model_id, "input": texts}


def parse_embedding_response(raw_json: dict) -> list[list[float]]:
    return [item["embedding"] for item in raw_json["data"]]


def embed_via_openrouter(settings: Settings, profile: EmbeddingProfile, texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json=build_embedding_request(profile, texts),
        timeout=60,
    )
    response.raise_for_status()
    return parse_embedding_response(response.json())


def get_embedder_for_profile(settings: Settings, profile: EmbeddingProfile):
    if profile.provider != "local":
        return None
    return TextEmbedding(model_name=profile.model_id, cache_dir=settings.model_cache_dir)


def embed_for_profile(
    settings: Settings, profile: EmbeddingProfile, embedder, texts: list[str]
) -> list[list[float]]:
    if profile.provider == "local":
        return embed(embedder, texts)
    return embed_via_openrouter(settings, profile, texts)
```

Важно: `httpx` добавляется новым импортом в начало файла вместе с уже существующими
импортами (`from fastembed import TextEmbedding`, `from kf.config import Settings`) — не
дублировать, если он там уже есть.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embeddings.py -v`
Expected: PASS (6/6 — 2 исходных + 4 новых).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/embeddings.py" "knowledge-factory/tests/test_embeddings.py"
git commit -m "Digital brain | Провайдер эмбеддинга через OpenRouter в kf/embeddings.py | V 1.4.43"
```

---

### Task 4: Утилиты Qdrant — `list_paths` и `point_id`

**Files:**
- Modify: `knowledge-factory/kf/store/qdrant_store.py` (добавить в конец файла)
- Test: `knowledge-factory/tests/test_qdrant_store.py` (добавить новые тесты)

**Interfaces:**
- Consumes: ничего нового.
- Produces: `list_paths(client: QdrantClient, collection: str) -> set[str]` (возвращает
  пустое множество, если коллекции ещё нет), `point_id(path: str, chunk_index: int) -> str`
  (детерминированный UUID5 — одинаковый для одного и того же `path`+`chunk_index` независимо
  от того, в какую коллекцию/профиль пишем). Используются в Task 5 (замена приватного
  `_point_id` в `kf/ingest.py`) и Task 6 (`kf/embedding_sync.py`).

- [ ] **Step 1: Write the failing tests**

Добавить в конец `knowledge-factory/tests/test_qdrant_store.py`:

```python
from kf.store.qdrant_store import list_paths, point_id


def test_list_paths_returns_empty_set_for_missing_collection(client):
    assert list_paths(client, "коллекция_которой_нет") == set()


def test_list_paths_returns_unique_paths(client, embedder):
    vector = embed(embedder, ["текст"])[0]
    upsert_chunks(
        client,
        COLLECTION,
        [
            {"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": "a.md", "chunk_index": 0, "text": "x"}},
            {"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": "a.md", "chunk_index": 1, "text": "y"}},
            {"id": str(uuid.uuid4()), "vector": vector, "payload": {"path": "b.md", "chunk_index": 0, "text": "z"}},
        ],
    )

    paths = list_paths(client, COLLECTION)

    assert paths == {"a.md", "b.md"}


def test_point_id_is_deterministic_for_same_path_and_chunk():
    assert point_id("a.md", 0) == point_id("a.md", 0)


def test_point_id_differs_for_different_chunk_index():
    assert point_id("a.md", 0) != point_id("a.md", 1)


def test_point_id_differs_for_different_path():
    assert point_id("a.md", 0) != point_id("b.md", 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_qdrant_store.py -v`
Expected: FAIL с `ImportError: cannot import name 'list_paths'` — существующие тесты
продолжают проходить.

- [ ] **Step 3: Implement**

Добавить в начало `knowledge-factory/kf/store/qdrant_store.py` (рядом с существующими
импортами) `import uuid`, затем добавить в конец файла:

```python
_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}:{chunk_index}"))


def list_paths(client: QdrantClient, collection: str) -> set[str]:
    if not client.collection_exists(collection):
        return set()

    paths: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_payload=["path"],
            with_vectors=False,
            offset=offset,
        )
        paths.update(p.payload["path"] for p in points)
        if offset is None:
            break
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_qdrant_store.py -v`
Expected: PASS. Если сигнатура `client.scroll(...)` в установленной версии `qdrant-client`
отличается (например, порядок возвращаемых значений) — ошибка будет конкретной
(`TypeError`/`ValueError`); проверить фактическую сигнатуру через
`uv run python -c "from qdrant_client import QdrantClient; help(QdrantClient.scroll)"` и
поправить вызов, сохранив сигнатуру `list_paths(client, collection) -> set[str]` неизменной.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/store/qdrant_store.py" "knowledge-factory/tests/test_qdrant_store.py"
git commit -m "Digital brain | kf/store/qdrant_store.py: list_paths + детерминированный point_id | V 1.4.44"
```

---

### Task 5: Интеграция профиля в `kf/api.py`, `kf/cli.py`, `kf/ingest.py`

**Files:**
- Modify: `knowledge-factory/kf/api.py` (весь файл)
- Modify: `knowledge-factory/kf/cli.py:1-30` (`_build_ingest_deps` и импорты)
- Modify: `knowledge-factory/kf/ingest.py` (замена `_point_id`/`_NAMESPACE` и вызова `embed`
  в `_store_text`, добавление поля `profile` в `IngestDeps`)
- Test: `knowledge-factory/tests/test_api.py`
- Test: `knowledge-factory/tests/test_ingest.py`

**Interfaces:**
- Consumes: `EMBEDDING_PROFILES`, `get_profile` из Task 1; `get_active_profile_name` из
  Task 2; `embed_for_profile`, `get_embedder_for_profile` из Task 3; `point_id` из Task 4.
- Produces: `KnowledgeSession.profile: EmbeddingProfile` (новое поле, по умолчанию профиль
  `local` — существующие вызовы `KnowledgeSession(...)` без этого аргумента продолжают
  работать); `IngestDeps.profile: EmbeddingProfile` (новое поле, тот же принцип). Используются
  в Task 6 и 7.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_api.py`:

```python
from kf.embedding_models import EMBEDDING_PROFILES


def test_open_session_defaults_to_local_profile():
    session = open_session()

    assert session.profile.name == "local"
    assert session.profile.collection == "knowledge"


def test_semantic_search_uses_session_profile_collection(monkeypatch):
    session = open_session()
    calls = []
    monkeypatch.setattr(
        "kf.api.qdrant_search",
        lambda client, collection, vector, limit: calls.append(collection) or [],
    )

    semantic_search(session, "тест", limit=1)

    assert calls == [session.profile.collection]
```

Добавить в `knowledge-factory/tests/test_ingest.py`:

```python
def test_ingest_uses_default_local_profile_when_not_specified(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Заметка про тест.", encoding="utf-8")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_ingested >= 1
    assert deps.profile.name == "local"


def test_ingest_calls_embed_for_profile_with_active_profile(tmp_path, deps, monkeypatch):
    from kf.embedding_models import EMBEDDING_PROFILES

    deps.profile = EMBEDDING_PROFILES["openai-small"]
    seen_profiles = []

    def _fake_embed_for_profile(settings, profile, embedder, texts):
        seen_profiles.append(profile.name)
        return [[0.0] * profile.dimension for _ in texts]

    monkeypatch.setattr("kf.ingest.embed_for_profile", _fake_embed_for_profile)
    (tmp_path / "note1.md").write_text("Заметка про тест профиля.", encoding="utf-8")

    ingest_directory(tmp_path, deps)

    assert "openai-small" in seen_profiles
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_api.py tests/test_ingest.py -k "profile" -v`
Expected: FAIL — `AttributeError: 'KnowledgeSession' object has no attribute 'profile'` и/или
`'IngestDeps' object has no attribute 'profile'`.

- [ ] **Step 3: Implement — `kf/api.py`**

Заменить содержимое `knowledge-factory/kf/api.py` целиком на:

```python
from dataclasses import dataclass, field

from kf.config import Settings, load_settings
from kf.embedding_models import EMBEDDING_PROFILES, EmbeddingProfile, get_profile
from kf.embedding_state import get_active_profile_name
from kf.embeddings import embed_for_profile, get_embedder_for_profile
from kf.llm import build_prompt, call_llm
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.graph_store import query_entity
from kf.store.postgres import connect, ensure_schema
from kf.store.qdrant_store import ensure_collection
from kf.store.qdrant_store import get_client as get_qdrant_client
from kf.store.qdrant_store import search as qdrant_search


@dataclass
class KnowledgeSession:
    settings: Settings
    pg_conn: object
    qdrant_client: object
    embedder: object
    graph_conn: object
    profile: EmbeddingProfile = field(default_factory=lambda: EMBEDDING_PROFILES["local"])


def open_session() -> KnowledgeSession:
    settings = load_settings()
    profile = get_profile(get_active_profile_name(settings.data_root))

    pg_conn = connect(settings)
    ensure_schema(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    ensure_collection(qdrant_client, profile.collection, vector_size=profile.dimension)

    embedder = get_embedder_for_profile(settings, profile)

    return KnowledgeSession(
        settings=settings,
        pg_conn=pg_conn,
        qdrant_client=qdrant_client,
        embedder=embedder,
        graph_conn=None,
        profile=profile,
    )


def semantic_search(session: KnowledgeSession, query: str, limit: int = 5) -> list[dict]:
    vector = embed_for_profile(session.settings, session.profile, session.embedder, [query])[0]
    results = qdrant_search(session.qdrant_client, session.profile.collection, vector, limit=limit)
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

    if session.qdrant_client.collection_exists(session.profile.collection):
        chunk_count = session.qdrant_client.get_collection(session.profile.collection).points_count
    else:
        chunk_count = 0

    return {"documents": doc_count, "chunks": chunk_count}


def graph_search(session: KnowledgeSession, entity: str) -> list[dict]:
    graph_conn = session.graph_conn
    owns_connection = graph_conn is None
    if owns_connection:
        graph_conn = get_graph_connection(session.settings)
        ensure_graph_schema(graph_conn)
    try:
        return query_entity(graph_conn, entity)
    finally:
        if owns_connection:
            del graph_conn
```

Обратить внимание: константы `COLLECTION`/`VECTOR_SIZE` больше не экспортируются из
`kf/api.py` — они заменены полем `session.profile`. `kf/cli.py` импортирует их напрямую
(см. следующий шаг) — это единственное место, которое их использовало.

- [ ] **Step 4: Implement — `kf/cli.py` (импорты и `_build_ingest_deps`)**

В `knowledge-factory/kf/cli.py` найти строку импорта:

```python
from kf.api import COLLECTION, VECTOR_SIZE, ask_question, get_stats, graph_search, open_session, semantic_search
```

Заменить на:

```python
from kf.api import ask_question, get_stats, graph_search, open_session, semantic_search
```

Найти функцию:

```python
def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    graph_conn = get_graph_connection(settings)
    ensure_graph_schema(graph_conn)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=COLLECTION,
        settings=settings,
        graph_conn=graph_conn,
    )
```

Заменить на:

```python
def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    graph_conn = get_graph_connection(settings)
    ensure_graph_schema(graph_conn)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=session.profile.collection,
        settings=settings,
        graph_conn=graph_conn,
        profile=session.profile,
    )
```

- [ ] **Step 5: Implement — `kf/ingest.py`**

В `knowledge-factory/kf/ingest.py` заменить весь блок от начала файла (импорты) до конца
функции `_store_text` включительно:

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
from kf.store.graph_store import add_relationship, delete_relationships_by_source, upsert_entity
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
```

на:

```python
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embedding_models import EMBEDDING_PROFILES, EmbeddingProfile
from kf.embeddings import embed_for_profile
from kf.extract import extract_text
from kf.graph import extract_entities_and_relationships
from kf.hashing import sha256_of_file
from kf.journal import append_entries, detect_deleted, extract_description, format_entry
from kf.scope import should_index
from kf.store.graph_store import add_relationship, delete_relationships_by_source, upsert_entity
from kf.store.minio_store import upload_file
from kf.store.postgres import list_paths, needs_ingest, path_known, record_ingested
from kf.store.qdrant_store import point_id, upsert_chunks
from kf.synthesize import synthesize_note


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
    profile: EmbeddingProfile = field(default_factory=lambda: EMBEDDING_PROFILES["local"])


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


def _store_text(text: str, rel_key: str, path: Path, deps: IngestDeps, stats: IngestStats) -> None:
    chunks = chunk_text(text, max_chars=deps.max_chars, overlap=deps.overlap)
    if chunks:
        vectors = embed_for_profile(deps.settings, deps.profile, deps.embedder, chunks)
        points = [
            {
                "id": point_id(rel_key, i),
                "vector": vectors[i],
                "payload": {"path": rel_key, "chunk_index": i, "text": chunks[i]},
            }
            for i in range(len(chunks))
        ]
        upsert_chunks(deps.qdrant_client, deps.collection, points)
        stats.chunks_written += len(points)
    upload_file(deps.minio_client, path, rel_key)
```

Ключевые изменения: `_NAMESPACE`/`_point_id` удалены целиком (переехали в
`kf/store/qdrant_store.py` в Task 4, используются через импорт `point_id`); `import uuid`
удалён (в файле после этой правки больше не используется); `embed` заменён на
`embed_for_profile` с новыми аргументами `deps.settings`/`deps.profile`; в `IngestDeps`
добавлено поле `profile` со значением по умолчанию — существующие вызовы `IngestDeps(...)`
без этого аргумента (`tests/test_ingest.py`) продолжают работать без изменений. Остальная
часть файла (`_synthesize_and_index_note`, `_extract_and_store_entities`,
`ingest_directory`) не меняется.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_api.py tests/test_ingest.py tests/test_cli.py -v`
Expected: PASS — все тесты, включая новые. Существующие тесты `test_ingest.py` не
модифицировались и должны проходить без изменений благодаря значению по умолчанию у
`IngestDeps.profile`.

- [ ] **Step 7: Commit**

```bash
git add "knowledge-factory/kf/api.py" "knowledge-factory/kf/cli.py" "knowledge-factory/kf/ingest.py" "knowledge-factory/tests/test_api.py" "knowledge-factory/tests/test_ingest.py"
git commit -m "Digital brain | Активный профиль эмбеддинга интегрирован в api/cli/ingest | V 1.4.45"
```

---

### Task 6: Логика синхронизации — `kf/embedding_sync.py`

**Files:**
- Create: `knowledge-factory/kf/embedding_sync.py`
- Test: `knowledge-factory/tests/test_embedding_sync.py`

**Interfaces:**
- Consumes: `EmbeddingProfile` из Task 1; `embed_for_profile` из Task 3; `point_id` из
  Task 4; `extract_text` (существующий, `kf.extract`); `chunk_text` (существующий,
  `kf.chunking`); `upsert_chunks` (существующий, `kf.store.qdrant_store`).
- Produces: `resolve_missing_path_text(rel_key, source_dir, notes_dir, settings) -> str | None`,
  `sync_missing_paths(missing_paths, source_dir, notes_dir, settings, profile, embedder, qdrant_client, collection=None) -> tuple[int, int]`
  (возвращает `(synced, failed)`; `collection` — необязательный override, по умолчанию
  `profile.collection` — тестам нужен отдельный от продакшена коллекции, продакшен-код
  в Task 7 его не передаёт). Используются в Task 7 (`kf.py embedding-model sync`).

- [ ] **Step 1: Write the failing tests**

Создать `knowledge-factory/tests/test_embedding_sync.py`:

```python
from pathlib import Path

import pytest

from kf.config import load_settings
from kf.embedding_models import EMBEDDING_PROFILES
from kf.embedding_sync import resolve_missing_path_text, sync_missing_paths
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
```

`sync_missing_paths` в тестах вызывается с `COLLECTION = "kf_test_sync"`, а не с
`profile.collection` — для этого `sync_missing_paths` должен принимать `profile` только для
разрешения провайдера/размерности эмбеддинга, а имя коллекции для записи — параметром
отдельно. Учтено в сигнатуре ниже.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embedding_sync.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.embedding_sync'`.

- [ ] **Step 3: Implement**

Создать `knowledge-factory/kf/embedding_sync.py`:

```python
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embedding_models import EmbeddingProfile
from kf.embeddings import embed_for_profile
from kf.extract import extract_text
from kf.store.qdrant_store import point_id, upsert_chunks

_MAX_CHARS = 1500
_OVERLAP = 150


def resolve_missing_path_text(
    rel_key: str, source_dir: Path, notes_dir: Path, settings: Settings
) -> str | None:
    notes_prefix = f"{notes_dir.name}/"
    if rel_key.startswith(notes_prefix):
        note_path = notes_dir.parent / rel_key
        if not note_path.exists():
            return None
        return note_path.read_text(encoding="utf-8")

    source_path = source_dir / rel_key
    if not source_path.exists():
        return None
    return extract_text(source_path, settings)


def sync_missing_paths(
    missing_paths: set[str],
    source_dir: Path,
    notes_dir: Path,
    settings: Settings,
    profile: EmbeddingProfile,
    embedder: object,
    qdrant_client: object,
    collection: str | None = None,
) -> tuple[int, int]:
    target_collection = collection or profile.collection
    synced = 0
    failed = 0

    for rel_key in sorted(missing_paths):
        try:
            text = resolve_missing_path_text(rel_key, source_dir, notes_dir, settings)
            if text is None:
                print(f"[embedding-sync] файл не найден на диске: {rel_key}")
                failed += 1
                continue

            chunks = chunk_text(text, max_chars=_MAX_CHARS, overlap=_OVERLAP)
            if not chunks:
                synced += 1
                continue

            vectors = embed_for_profile(settings, profile, embedder, chunks)
            points = [
                {
                    "id": point_id(rel_key, i),
                    "vector": vectors[i],
                    "payload": {"path": rel_key, "chunk_index": i, "text": chunks[i]},
                }
                for i in range(len(chunks))
            ]
            upsert_chunks(qdrant_client, target_collection, points)
            synced += 1
        except Exception as exc:
            print(f"[embedding-sync] не удалось синхронизировать {rel_key}: {exc}")
            failed += 1

    return synced, failed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_embedding_sync.py -v`
Expected: PASS (6/6).

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/embedding_sync.py" "knowledge-factory/tests/test_embedding_sync.py"
git commit -m "Digital brain | Логика синхронизации недостающих эмбеддингов: kf/embedding_sync.py | V 1.4.46"
```

---

### Task 7: Команды `kf.py embedding-model list/use/sync`

**Files:**
- Modify: `knowledge-factory/kf/cli.py` (добавить импорты и новую группу команд)
- Test: `knowledge-factory/tests/test_cli.py`

**Interfaces:**
- Consumes: всё из Task 1, 2, 4, 6; `list_paths` (Postgres, существующий, `kf.store.postgres`);
  `detect_deleted` (существующий, `kf.journal`) — переиспользуется для расчёта "не хватает"
  как `detect_deleted(known_paths, indexed_paths)`.
- Produces: CLI-команды `kf.py embedding-model list`, `kf.py embedding-model use <name>`,
  `kf.py embedding-model sync`.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_cli.py`:

```python
def test_embedding_model_list_marks_active_profile(monkeypatch):
    monkeypatch.setattr("kf.cli.get_active_profile_name", lambda data_root: "local")
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: set())
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: set())
    monkeypatch.setattr("kf.cli.connect", lambda settings: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda settings: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "list"])

    assert result.exit_code == 0
    assert "* local" in result.output


def test_embedding_model_use_rejects_unknown_profile():
    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "use", "не-существует"])

    assert result.exit_code != 0
    assert "неизвестный профиль" in result.output


def test_embedding_model_use_switches_and_warns_about_gap(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: {"a.md", "b.md"})
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: {"a.md"})
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "use", "qwen3-8b"])

    assert result.exit_code == 0
    assert "Активная модель: qwen3-8b" in result.output
    assert "не хватает 1" in result.output


def test_embedding_model_sync_reports_up_to_date_when_nothing_missing(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.get_active_profile_name", lambda data_root: "local")
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: {"a.md"})
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: {"a.md"})
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_collection", lambda client, collection, vector_size: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "sync"])

    assert result.exit_code == 0
    assert "актуальна" in result.output


def test_embedding_model_sync_calls_sync_missing_paths(monkeypatch, tmp_path):
    settings = load_settings()
    settings.data_root = str(tmp_path)
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.get_active_profile_name", lambda data_root: "local")
    monkeypatch.setattr("kf.cli.list_paths", lambda pg_conn: {"a.md", "b.md"})
    monkeypatch.setattr("kf.cli.qdrant_list_paths", lambda client, collection: {"a.md"})
    monkeypatch.setattr("kf.cli.connect", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_schema", lambda conn: None)
    monkeypatch.setattr("kf.cli.get_qdrant_client", lambda s: None)
    monkeypatch.setattr("kf.cli.ensure_collection", lambda client, collection, vector_size: None)
    monkeypatch.setattr("kf.cli.get_embedder_for_profile", lambda settings, profile: None)
    seen = {}

    def _fake_sync(missing_paths, source_dir, notes_dir, settings, profile, embedder, qdrant_client):
        seen["missing"] = missing_paths
        return (1, 0)

    monkeypatch.setattr("kf.cli.sync_missing_paths", _fake_sync)

    runner = CliRunner()
    result = runner.invoke(cli, ["embedding-model", "sync"])

    assert result.exit_code == 0
    assert seen["missing"] == {"b.md"}
    assert "синхронизировано: 1" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_cli.py -k embedding_model -v`
Expected: FAIL — `Error: No such command 'embedding-model'`.

- [ ] **Step 3: Implement**

В `knowledge-factory/kf/cli.py` заменить блок импортов в начале файла:

```python
from pathlib import Path

import click

from kf.api import ask_question, get_stats, graph_search, open_session, semantic_search
from kf.config import load_settings
from kf.ingest import IngestDeps, ingest_directory
from kf.store.graph_store import ensure_schema as ensure_graph_schema
from kf.store.graph_store import get_connection as get_graph_connection
from kf.store.minio_store import ensure_bucket
from kf.store.minio_store import get_client as get_minio_client

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "raw-data-repository"
```

на:

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

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent.parent / "raw-data-repository"
```

Добавить в конец `knowledge-factory/kf/cli.py`, перед `if __name__ == "__main__":`:

```python
@cli.group(name="embedding-model")
def embedding_model():
    """Управление активной моделью эмбеддинга."""


@embedding_model.command(name="list")
def embedding_model_list():
    """Показать все профили моделей и их покрытие относительно текущей базы."""
    settings = load_settings()
    active = get_active_profile_name(settings.data_root)

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    known_paths = list_paths(pg_conn)

    qdrant_client = get_qdrant_client(settings)

    for name, profile in EMBEDDING_PROFILES.items():
        indexed_paths = qdrant_list_paths(qdrant_client, profile.collection)
        missing = detect_deleted(known_paths, indexed_paths)
        marker = "* " if name == active else "  "
        click.echo(
            f"{marker}{name} ({profile.provider}, {profile.model_id}, размерность={profile.dimension}): "
            f"{len(indexed_paths)}/{len(known_paths)} файлов, не хватает {len(missing)}"
        )


@embedding_model.command(name="use")
@click.argument("name")
def embedding_model_use(name: str):
    """Переключить активную модель эмбеддинга (без переиндексации)."""
    settings = load_settings()
    try:
        profile = get_profile(name)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    set_active_profile_name(settings.data_root, name)

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    known_paths = list_paths(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    indexed_paths = qdrant_list_paths(qdrant_client, profile.collection)
    missing = detect_deleted(known_paths, indexed_paths)

    click.echo(f"Активная модель: {name}")
    if missing:
        click.echo(
            f"⚠ Коллекция отстаёт: не хватает {len(missing)} файлов. "
            f"Запустите 'kf.py embedding-model sync', чтобы досчитать."
        )


@embedding_model.command(name="sync")
def embedding_model_sync():
    """Досчитать эмбеддинги только для файлов, которых не хватает в активной коллекции."""
    settings = load_settings()
    profile = get_profile(get_active_profile_name(settings.data_root))

    pg_conn = connect(settings)
    ensure_schema(pg_conn)
    known_paths = list_paths(pg_conn)

    qdrant_client = get_qdrant_client(settings)
    ensure_collection(qdrant_client, profile.collection, vector_size=profile.dimension)
    indexed_paths = qdrant_list_paths(qdrant_client, profile.collection)
    missing = detect_deleted(known_paths, indexed_paths)

    if not missing:
        click.echo("Коллекция уже актуальна, синхронизация не нужна.")
        return

    embedder = get_embedder_for_profile(settings, profile)
    notes_dir = Path(settings.synthesis_notes_dir)
    synced, failed = sync_missing_paths(
        missing, DEFAULT_SOURCE, notes_dir, settings, profile, embedder, qdrant_client
    )
    click.echo(f"Готово. синхронизировано: {synced}, ошибок: {failed}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_cli.py -v`
Expected: PASS — все тесты файла, включая пять новых.

Затем прогнать полный набор тестов:

Run: `cd "knowledge-factory" && uv run python -m pytest -q`
Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/cli.py" "knowledge-factory/tests/test_cli.py"
git commit -m "Digital brain | Команды kf.py embedding-model list/use/sync | V 1.4.47"
```

---

### Task 8: Документация

**Files:**
- Modify: `knowledge-factory/README.md`
- Modify: `карта_бз.md`

Задача без тестов (документация) — TDD не применяется, отдельный шаг с коммитом.

- [ ] **Step 1: Обновить `knowledge-factory/README.md`**

Найти блок:

```markdown
## Команды kf.py

```
uv run python kf.py ingest [--source PATH]   # индексация (по умолчанию — ../raw-data-repository)
uv run python kf.py search "запрос"          # семантический поиск фрагментов
uv run python kf.py ask "вопрос"             # поиск + связный ответ через OpenRouter, со ссылками
uv run python kf.py stats                    # сколько документов/чанков в базе
```
```

Заменить на:

```markdown
## Команды kf.py

```
uv run python kf.py ingest [--source PATH]   # индексация (по умолчанию — ../raw-data-repository)
uv run python kf.py search "запрос"          # семантический поиск фрагментов
uv run python kf.py ask "вопрос"             # поиск + связный ответ через OpenRouter, со ссылками
uv run python kf.py stats                    # сколько документов/чанков в базе
uv run python kf.py embedding-model list     # профили моделей эмбеддинга и их покрытие
uv run python kf.py embedding-model use <n>  # переключить активную модель (без переиндексации)
uv run python kf.py embedding-model sync     # досчитать недостающие эмбеддинги для активной модели
```
```

В конец файла, перед разделом "## Разработка (uv)", добавить новый раздел:

```markdown
## Переключаемые модели эмбеддинга

Доступны четыре профиля (`knowledge-factory/kf/embedding_models.py`): `local` (текущая
локальная MiniLM, без сети), `qwen3-8b`, `openai-small`, `openai-large` (через OpenRouter,
используют уже настроенный `OPENROUTER_API_KEY`). У каждого профиля своя коллекция в Qdrant —
переключение на уже использовавшийся профиль мгновенное и бесплатное.

`kf.py embedding-model use <имя>` только переключает активный профиль (файл состояния
`data/active_embedding_model.txt`) и предупреждает, если у выбранной модели не хватает
файлов. `kf.py embedding-model sync` — единственная команда, которая тратит API-бюджет (если
активная модель не `local`): досчитывает эмбеддинги только для отсутствующих файлов, не
трогая LLM-синтез заметок и граф знаний.
```

- [ ] **Step 2: Обновить `карта_бз.md`**

В самом конце файла, после раздела "## Граф знаний (сущности и связи)" и перед "## История
переименований и переносов", добавить новый раздел:

```markdown
## Переключаемые модели эмбеддинга

Четыре профиля модели эмбеддинга (`local`, `qwen3-8b`, `openai-small`, `openai-large`) — у
каждого своя коллекция в Qdrant, переключение на уже использовавшийся профиль мгновенное.
`kf.py embedding-model list/use/sync`: `use` переключает и бесплатно проверяет покрытие,
`sync` — единственная команда, которая реально тратит API-бюджет OpenRouter, досчитывая
только недостающие файлы.
```

- [ ] **Step 3: Commit**

```bash
git add "knowledge-factory/README.md" "карта_бз.md"
git commit -m "Digital brain | Документация переключаемых моделей эмбеддинга | V 1.4.48"
```
