from mcp.server.fastmcp import FastMCP

from kf.api import KnowledgeSession
from kf.api import ask_question as _ask_question
from kf.api import get_stats as _get_stats
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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
