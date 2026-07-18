# knowledge-factory — Knowledge Factory

Технический бэкенд семантического поиска для проекта Digital Brain: Docker-стек
(Qdrant + Postgres/pgvector + MinIO + Redis, профиль LIGHT) + CLI `kf.py` +
Knowledge MCP-сервер.

Полная архитектура и профили — `../project-config/Архитектура проекта. Настройка и Правила/core.md`,
`rules.md`, `ПЕРЕД-СТАРТОМ.md`. Текущий статус и история решений —
`../project-config/РЕШЕНИЯ-И-СТАТУС.md`.

## Статус: всё работает

- Все 4 контейнера healthy: Qdrant, Postgres, MinIO, Redis.
- `kf.py` (ingest/search/ask/stats) — готов, 74 тест (TDD), включая
  автоматический слой синтез-заметок поверх обычной индексации.
- Knowledge MCP-сервер — зарегистрирован в Claude Code, `claude mcp list` → Connected.

## Запуск стека

```
docker compose up -d
docker compose ps   # дождаться healthy у всех 4
```

- Qdrant: http://localhost:6333/dashboard
- MinIO: http://localhost:9001
- Postgres: localhost:5432
- Redis: localhost:**6380** (не 6379 — порт занят чужим контейнером другого проекта)

Тома данных — `data/` внутри этой же папки (физически на `E:`). Postgres — на именованном
томе Docker с фиксированным именем `knowledge-factory_kf_postgres_data` (не в `data/` —
Windows/NTFS bind-mount не даёт контейнеру строгих Unix-прав, которые требует Postgres; имя
тома зафиксировано явно в `docker-compose.yml`, чтобы не зависеть от имени папки проекта).
`data/` — в `.gitignore` (большие бинарники, не для git).

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

Исключено из индексации: `Закладки браузера — структура.md` (низкая ценность для поиска,
несоразмерный объём) — см. `kf/scope.py` (`EXCLUDED_FILENAMES`).

`kf.py ingest` теперь дополнительно пишет для каждого проиндексированного
файла осмысленную LLM-заметку в `Синтезированные данные (synthesized-notes)/`
(рядом с `data/`) и сразу делает её доступной для `search`/`ask`.

Каждый запуск `kf.py ingest` также пополняет `../Журнал знаний.md` (в корне проекта) —
короткую запись на каждое добавление/изменение/удаление файла в `raw-data-repository/`,
включая изменения, сделанные вручную в Obsidian. Удалённые файлы только логируются —
их векторы/записи в Qdrant/Postgres/MinIO не удаляются автоматически.

`kf.py ingest` также извлекает сущности (люди, инструменты, проекты, темы, концепты) и
связи между ними в граф знаний (Kuzu, `data/graph/`, без Docker). Посмотреть прямые связи
конкретной сущности: `uv run python kf.py graph "название сущности"`, или через MCP
(`graph_search`). Сбой извлечения не прерывает `ingest` — только логируется.

## Knowledge MCP

`kf/mcp_server.py` (FastMCP) поверх `kf/api.py` — инструменты `semantic_search`, `ask`,
`stats`, `graph_search`.

```
claude mcp add knowledge-factory -s user -e PYTHONPATH="E:\Digital brain\knowledge-factory" -- "E:\Digital brain\knowledge-factory\.venv\Scripts\python.exe" -m kf.mcp_server
```

(Имя сервера в Claude Code — `knowledge-factory`, совпадает с именем папки.)
Нужна новая сессия/перезапуск Claude Code, чтобы инструменты стали доступны в разговоре.

## Тесты

```
uv run pytest
```

74 тест: часть — чистые юнит-тесты (scope/chunking/hashing/extract/llm), часть — интеграционные
против живого стека (Postgres/Qdrant/MinIO/MCP-протокол), без моков.

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

## Разработка (uv)

```
uv sync              # установить/пересоздать .venv
uv add <пакет>        # добавить зависимость
```

При переносе папки проекта на другой диск/путь — `.venv` пересоздавать заново (`uv sync`),
не копировать: внутри зашиты абсолютные пути.
