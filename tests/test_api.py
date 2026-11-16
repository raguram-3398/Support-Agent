import json

import httpx
import pytest
from conftest import make_token, serve_test_jwks
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import api
from support_agent.api import API_AUDIENCE, app
from support_agent.graph import build_graph

ALICE, BOB, LARGE_ORDER = 1, 2, 2


def auth(customer_id: int) -> dict:
    token = make_token(API_AUDIENCE, customer_id=customer_id)
    return {"Authorization": f"Bearer {token}"}


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def ainvoke(self, messages):
        return next(self.replies)


LARGE_REFUND = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "issue_refund",
            "args": {"order_id": LARGE_ORDER, "reason": "broken"},
            "id": "r1",
        }
    ],
)


@pytest.fixture
def client(fake_mcp, monkeypatch):
    serve_test_jwks(monkeypatch, api.verifier)
    model = ScriptedModel([LARGE_REFUND, AIMessage("refunded")])
    app.state.graph = build_graph(model, InMemorySaver(), fake_mcp[0])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://api")


def events(response: httpx.Response) -> list[tuple[str, object]]:
    parsed = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


CHAT = {"thread_id": "t1", "message": "refund order 2"}


async def test_chat_streams_until_approval_then_approve_finishes(client, fake_mcp):
    async with client:
        chat = await client.post("/chat", json=CHAT, headers=auth(ALICE))
        assert chat.headers["content-type"].startswith("text/event-stream")
        assert [name for name, _ in events(chat)] == [
            "agent",
            "approval_required",
            "done",
        ]
        assert events(chat)[1][1] == {
            "refunds": [{"order_id": LARGE_ORDER, "total_cents": 15000}]
        }
        assert fake_mcp[2] == []

        resumed = await client.post(
            "/approve", json={"thread_id": "t1", "approved": True}, headers=auth(ALICE)
        )
        assert [name for name, _ in events(resumed)] == ["tools", "agent", "done"]
        assert fake_mcp[2] == [(ALICE, LARGE_ORDER)]


async def test_approve_on_thread_not_paused_is_409(client):
    async with client:
        response = await client.post(
            "/approve",
            json={"thread_id": "never-started", "approved": True},
            headers=auth(ALICE),
        )
    assert response.status_code == 409


async def test_chat_on_paused_thread_is_409(client):
    async with client:
        await client.post("/chat", json=CHAT, headers=auth(ALICE))
        response = await client.post("/chat", json=CHAT, headers=auth(ALICE))
    assert response.status_code == 409


async def test_chat_without_token_is_401(client):
    async with client:
        response = await client.post("/chat", json=CHAT)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_chat_with_mcp_audience_token_is_401(client):
    token = make_token("http://127.0.0.1:8000/mcp", customer_id=ALICE)
    async with client:
        response = await client.post(
            "/chat", json=CHAT, headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 401


async def test_token_without_customer_id_is_401(client):
    token = make_token(API_AUDIENCE)
    async with client:
        response = await client.post(
            "/chat", json=CHAT, headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 401


async def test_customer_id_in_body_is_ignored(client, fake_mcp):
    async with client:
        await client.post(
            "/chat", json={**CHAT, "customer_id": BOB}, headers=auth(ALICE)
        )
    assert fake_mcp[1] == [(ALICE, LARGE_ORDER)]


async def test_other_customer_cannot_use_the_thread(client, fake_mcp):
    async with client:
        await client.post("/chat", json=CHAT, headers=auth(ALICE))
        chat = await client.post("/chat", json=CHAT, headers=auth(BOB))
        approve = await client.post(
            "/approve", json={"thread_id": "t1", "approved": True}, headers=auth(BOB)
        )
    assert chat.status_code == 404
    assert approve.status_code == 404
    assert fake_mcp[2] == []
