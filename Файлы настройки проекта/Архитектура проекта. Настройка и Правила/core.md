Core:
- Qdrant
- PostgreSQL
- MinIO
- Redis

Advanced:
- OpenSearch
- Neo4j или Kuzu
- LlamaIndex
- LangGraph
- LiteLLM
- Mem0 / Letta для agent memory

Метод:
Hybrid RAG + GraphRAG + reranking + RBAC-aware retrieval


AiSpase Home Knowledge Fabric
├── MinIO
│   └── исходные документы, файлы, артефакты
│
├── Obsidian Vault
│   └── человекочитаемые знания, ADR, заметки, решения
│
├── PostgreSQL + pgvector
│   └── metadata, документы, chunks, статусы индексации
│
├── Qdrant
│   └── vector search, semantic search, hybrid search
│
├── Reranker
│   └── пересортировка найденных фрагментов
│
├── Neo4j
│   └── граф сущностей, связей, решений, компонентов
│
├── GraphRAG Worker
│   └── построение графа и multi-hop retrieval
│
└── Knowledge MCP
    └── доступ агентов к semantic_search, hybrid_search, graph_search


=====================================================================
ПРЕДПОДГОТОВКА И ТРЕБОВАНИЯ (ЧИТАТЬ ПЕРЕД ЗАПУСКОМ)
=====================================================================
Этот блок предотвращает типовые ошибки на разном железе (Windows / Mac /
Linux, мало RAM, забит диск). Выбери профиль под СВОЁ железо — иначе сервисы
не влезут в память и всё будет свопиться или падать по OOM.

--- ПРОФИЛИ РАЗВЁРТЫВАНИЯ ПО RAM ---

LIGHT (8–16 GB RAM) — рекомендуется по умолчанию, ~4–8 GB потребления:
- Qdrant            → vector + HYBRID search (заменяет OpenSearch)
- PostgreSQL+pgvector → metadata, chunks, статусы индексации
- MinIO             → raw-файлы
- Redis             → кэш / short-term memory
- Kuzu              → граф, ВСТРОЕННЫЙ, без JVM (вместо Neo4j)
- Reranker          → API (Jina / Cohere) или лёгкий bge-reranker-base
- Embeddings        → API через LiteLLM (НЕ локальная модель)
- Memory            → Mem0 (lite)
БЕЗ OpenSearch, БЕЗ Neo4j-JVM, БЕЗ тяжёлых локальных моделей.

FULL (32 GB+ RAM) — ~16–24 GB потребления:
- + OpenSearch (sparse / BM25)
- + Neo4j (вместо Kuzu)
- + локальные embeddings (BGE/Qwen) и локальный cross-encoder reranker
- + GraphRAG Worker на полную

ПРАВИЛО: OpenSearch и Neo4j — самые прожорливые (JVM). На < 32 GB их НЕ включать.

--- РАЗМЕЩЕНИЕ ДАННЫХ (КРИТИЧНО) ---
Тома данных (MinIO, Qdrant, Postgres, граф) размещать на ОТДЕЛЬНОМ быстром
диске с запасом ≥ 50 GB свободно, НЕ на системном. Bind-mount всех томов в
одну папку данных:
- Mac/Linux: /<быстрый_диск>/knowledge-factory/data/{minio,qdrant,postgres,graph,redis}
- Windows:   D:\knowledge-factory\data\...   (не внутри забитого C:)
Системный диск держать со свободными ≥ 10 GB под образы Docker.

--- КРОСС-ПЛАТФОРМА (Docker) ---
- Windows: Docker Desktop + WSL2 (включить WSL2 backend). В %UserProfile%\.wslconfig
  ограничить память: [wsl2] memory=12GB — чтобы WSL2 не съел всю систему.
- macOS: OrbStack (легче) или Docker Desktop. В Resources поднять RAM-лимит
  не ниже профиля; Disk image при нехватке системного диска перенести на внешний.
- Linux: нативный Docker Engine + docker compose plugin.

--- ЧТО ИНДЕКСИРОВАТЬ (RAW SCOPE) ---
Индексировать ТОЛЬКО текстовые знания. ИСКЛЮЧИТЬ из ingest:
- видео / аудио / большие медиа, бинарники
- node_modules, .git, venv, dist, build, кэши
- архивы целиком
Иначе индексация долгая и забивает диск бесполезными эмбеддингами.
# КАК ЭТО РАБОТАЕТ (общая схема)

Фабрика знаний превращает разрозненные файлы в умный поиск по смыслу:
1. RAW — твои файлы (лекции, кейсы, документы) лежат на SSD.
2. INGEST — `kf.py ingest` читает их, режет на фрагменты (чанки), считает эмбеддинги
   ЛОКАЛЬНО (приватно) и кладёт: векторы → Qdrant, метаданные/текст → Postgres, сырьё → MinIO.
3. ЗАПРОС — ты спрашиваешь по-русски обычным языком.
4. RETRIEVAL — запрос превращается в вектор, Qdrant находит ближайшие по смыслу фрагменты
   (+ Postgres по ключевым словам = гибрид).
5. SYNTHESIS — найденные фрагменты уходят в LLM (OpenRouter), он собирает связный ответ
   СО ССЫЛКАМИ на источники. Наружу уходит только горстка найденных кусков, не вся база.

# РАЗВЁРТЫВАНИЕ ПОД КЛЮЧ (полный конвейер одной серией, не по кускам)
    docker compose up -d                 # 1. инфра (ждать healthy: qdrant, postgres, minio, redis)
    kf.py ingest                         # 2. индексация: векторы→Qdrant, текст→Postgres (локально)
    kf.py upload                         # 3. сырые файлы→MinIO (бакет raw)
    claude mcp add knowledge-factory ... # 4. подключить MCP к Claude/боту
    kf.py stats                          # 5. проверка: документы/чанки > 0 + тестовый ask
Всё идемпотентно (resume по sha256/размеру) — повтор безопасен, не дублирует.

# КАК ПОЛЬЗОВАТЬСЯ
- Через MCP (в Claude/боте): «спроси базу знаний: ...» → инструменты mcp__knowledge-factory__*.
- Через терминал из /Volumes/SSD/knowledge-factory:
    kf.py ask "вопрос"     — ответ + источники
    kf.py search "запрос"  — найти фрагменты
    kf.py ingest           — дозалить новые файлы (resume, не дублирует)
    kf.py upload           — дозалить сырые файлы в MinIO (resume)
    kf.py stats            — сколько в базе
- Добавил новые файлы в источники → `kf.py ingest && kf.py upload`, обработаются только новые.

# КАК ПРОВЕРИТЬ ЧТО РАБОТАЕТ
- Контейнеры: OrbStack → Containers (зелёные) или `docker ps` (4 шт. healthy).
- Векторы глазами: http://localhost:6333/dashboard (коллекция knowledge).
- Файлы-сырьё: http://localhost:9001 (MinIO).
- Быстрый тест: `kf.py stats` → должны быть документы и чанки > 0.
