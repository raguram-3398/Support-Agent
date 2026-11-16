import os

import httpx
import pytest
from conftest import make_token, serve_test_jwks
from mcp import Client

from support_agent import mcp_server
from support_agent.mcp_server import RESOURCE_URL, http_app, mcp

needs_db = pytest.mark.skipif(
    os.environ.get("CI_SKIP_DB") == "1", reason="no Postgres in CI"
)

ALICE, ALICE_BIG_ORDER = 1, 2
HOST = "127.0.0.1:8000"
PROTOCOL = "2026-07-28"
META = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL,
    "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
LIST_TOOLS = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {"_meta": META},
}
REFUND = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "_meta": META,
        "name": "issue_refund",
        "arguments": {"customer_id": ALICE, "order_id": ALICE_BIG_ORDER, "reason": "x"},
    },
}
BOTH_SCOPES = "orders:read refunds:write"


@pytest.fixture(autouse=True)
def test_keys(monkeypatch):
    serve_test_jwks(monkeypatch, mcp_server.verifier)


def agent_token(scope: str = BOTH_SCOPES, audience: str = RESOURCE_URL) -> str:
    return make_token(audience, azp="support-agent", scope=scope)


async def post(body: dict, token: str | None, host: str = HOST, name: str = ""):
    app = http_app()
    headers = {
        "Host": host,
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL,
        "Mcp-Method": body["method"],
    }
    if name:
        headers["Mcp-Name"] = name
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("http://mcp/mcp", json=body, headers=headers)


async def test_no_token_is_401_pointing_to_metadata():
    response = await post(LIST_TOOLS, token=None)
    assert response.status_code == 401
    challenge = response.headers["www-authenticate"]
    assert challenge.startswith("Bearer ")
    assert (
        'resource_metadata="http://127.0.0.1:8000/.well-known/oauth-protected-resource/mcp"'
        in challenge
    )


async def test_protected_resource_metadata_names_keycloak():
    app = http_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get(
            "http://mcp/.well-known/oauth-protected-resource/mcp",
            headers={"Host": HOST},
        )
    assert response.status_code == 200
    assert response.json()["resource"] == RESOURCE_URL
    assert response.json()["authorization_servers"] == [
        "http://127.0.0.1:8080/realms/support"
    ]


async def test_agent_token_lists_the_three_tools():
    response = await post(LIST_TOOLS, agent_token())
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert names == {"get_order", "list_orders", "issue_refund"}


async def test_user_token_for_the_api_is_rejected():
    response = await post(LIST_TOOLS, make_token("support-api", customer_id=ALICE))
    assert response.status_code == 401


async def test_token_without_read_scope_is_403():
    response = await post(LIST_TOOLS, agent_token(scope="refunds:write"))
    assert response.status_code == 403
    assert 'error="insufficient_scope"' in response.headers["www-authenticate"]


async def test_read_only_token_gets_step_up_challenge_on_refund():
    response = await post(REFUND, agent_token(scope="orders:read"), name="issue_refund")
    assert response.status_code == 403
    challenge = response.headers["www-authenticate"]
    assert 'error="insufficient_scope"' in challenge
    assert 'scope="refunds:write"' in challenge


async def test_refund_hidden_behind_a_wrong_name_header_is_refused():
    response = await post(REFUND, agent_token(scope="orders:read"), name="get_order")
    assert response.status_code == 400


async def test_tool_refuses_refund_without_write_scope_even_past_the_gate():
    async with Client(mcp) as client:
        result = await client.call_tool("issue_refund", REFUND["params"]["arguments"])
    assert result.is_error is True
    assert "refunds:write" in result.content[0].text


async def test_http_rejects_unknown_host_header():
    response = await post(LIST_TOOLS, agent_token(), host="evil.example:8000")
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
