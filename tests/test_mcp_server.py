import os

import httpx
import pytest
from mcp import Client

from support_agent.mcp_server import http_app, mcp

needs_db = pytest.mark.skipif(
    os.environ.get("CI_SKIP_DB") == "1", reason="no Postgres in CI"
)

ALICE, ALICE_BIG_ORDER = 1, 2
PROTOCOL = "2026-07-28"
LIST_TOOLS = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": PROTOCOL,
            "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    },
}


async def post_list_tools(host: str) -> httpx.Response:
    app = http_app()
    headers = {
        "Host": host,
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL,
        "Mcp-Method": "tools/list",
    }
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("http://mcp/mcp", json=LIST_TOOLS, headers=headers)


async def test_http_lists_the_three_tools():
    response = await post_list_tools("127.0.0.1:8000")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert names == {"get_order", "list_orders", "issue_refund"}


async def test_http_rejects_unknown_host_header():
    response = await post_list_tools("evil.example:8000")
    assert response.status_code == 421


@needs_db
async def test_get_order_through_mcp(no_refunds):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_order", {"customer_id": ALICE, "order_id": ALICE_BIG_ORDER}
        )
    assert result.is_error is False
    assert result.structured_content["result"]["total_cents"] == 15000


async def test_bad_arguments_return_error_result():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_order", {"customer_id": ALICE, "order_id": "abc"}
        )
    assert result.is_error is True
