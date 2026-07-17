import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from kf.mcp_server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_lists_expected_tools():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.list_tools()
        names = {t.name for t in result.tools}

    assert {"semantic_search", "ask", "stats"} <= names


@pytest.mark.anyio
async def test_stats_tool_returns_document_and_chunk_counts():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("stats", {})

    payload = json.loads(result.content[0].text)
    assert "documents" in payload
    assert "chunks" in payload


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

    assert result.structuredContent == {"result": []}
