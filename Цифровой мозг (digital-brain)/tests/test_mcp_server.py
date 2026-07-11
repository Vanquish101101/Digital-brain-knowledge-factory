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
