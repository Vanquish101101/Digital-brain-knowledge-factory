# Цифровой мозг (digital-brain) — Knowledge Factory

Технический бэкенд семантического поиска для проекта Digital Brain: Docker-стек
(Qdrant + Postgres/pgvector + MinIO + Redis, профиль LIGHT) + CLI `kf.py` +
Knowledge MCP-сервер.

Полная архитектура и профили — `../Файлы настройки проекта/Архитектура проекта. Настройка и Правила/core.md`,
`rules.md`, `ПЕРЕД-СТАРТОМ.md`. Текущий статус и история решений —
`../Файлы настройки проекта/РЕШЕНИЯ-И-СТАТУС.md`.

## Статус: всё работает

- Все 4 контейнера healthy: Qdrant, Postgres, MinIO, Redis.
- `kf.py` (ingest/search/ask/stats) — готов, 51 тест (TDD).
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
uv run python kf.py ingest [--source PATH]   # индексация (по умолчанию — ../Хранилище входных данных)
uv run python kf.py search "запрос"          # семантический поиск фрагментов
uv run python kf.py ask "вопрос"             # поиск + связный ответ через OpenRouter, со ссылками
uv run python kf.py stats                    # сколько документов/чанков в базе
```

Исключено из индексации: `Закладки браузера — структура.md` (низкая ценность для поиска,
несоразмерный объём) — см. `kf/scope.py` (`EXCLUDED_FILENAMES`).

## Knowledge MCP

`kf/mcp_server.py` (FastMCP) поверх `kf/api.py` — инструменты `semantic_search`, `ask`, `stats`.

```
claude mcp add knowledge-factory -s user -e PYTHONPATH="E:\Digital brain\Цифровой мозг (digital-brain)" -- "E:\Digital brain\Цифровой мозг (digital-brain)\.venv\Scripts\python.exe" -m kf.mcp_server
```

(Имя сервера в Claude Code — `knowledge-factory`, не переименовывалось вместе с папкой.)
Нужна новая сессия/перезапуск Claude Code, чтобы инструменты стали доступны в разговоре.

## Тесты

```
uv run pytest
```

51 тест: часть — чистые юнит-тесты (scope/chunking/hashing/extract/llm), часть — интеграционные
против живого стека (Postgres/Qdrant/MinIO/MCP-протокол), без моков.

## Разработка (uv)

```
uv sync              # установить/пересоздать .venv
uv add <пакет>        # добавить зависимость
```

При переносе папки проекта на другой диск/путь — `.venv` пересоздавать заново (`uv sync`),
не копировать: внутри зашиты абсолютные пути.
