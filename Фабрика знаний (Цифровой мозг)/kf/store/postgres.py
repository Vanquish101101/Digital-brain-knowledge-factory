import psycopg

from kf.config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                sha256 TEXT NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def needs_ingest(conn: psycopg.Connection, path: str, sha256: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT sha256 FROM documents WHERE path = %s", (path,))
        row = cur.fetchone()
    if row is None:
        return True
    return row[0] != sha256


def record_ingested(conn: psycopg.Connection, path: str, sha256: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (path, sha256, ingested_at)
            VALUES (%s, %s, now())
            ON CONFLICT (path) DO UPDATE
                SET sha256 = EXCLUDED.sha256,
                    ingested_at = EXCLUDED.ingested_at
            """,
            (path, sha256),
        )
    conn.commit()
