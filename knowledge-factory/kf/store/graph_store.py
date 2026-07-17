from pathlib import Path

import kuzu

from kf.config import Settings


def normalize_entity_name(name: str) -> str:
    return name.strip().lower()


def get_connection(settings: Settings) -> kuzu.Connection:
    graph_dir = Path(settings.data_root) / "graph"
    graph_dir.parent.mkdir(parents=True, exist_ok=True)
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
