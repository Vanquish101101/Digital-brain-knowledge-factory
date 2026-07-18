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
