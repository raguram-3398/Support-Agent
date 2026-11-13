from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from mcp.server import MCPServer

from support_agent.graph import MAX_STEPS, build_graph

ALICE, BOB_ORDER = 1, 4


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        return next(self.replies)


class LoopingModel:
    async def ainvoke(self, messages):
        return tool_call("get_order", {"order_id": 1}, f"call_{len(messages)}")


def tool_call(name, args, call_id):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


@pytest.fixture
def fake_mcp():
    server = MCPServer("fake")
    calls = []

    @server.tool()
    async def get_order(customer_id: int, order_id: int) -> dict[str, Any] | None:
        calls.append((customer_id, order_id))
        return {"order_id": order_id, "status": "delivered"}

    @server.tool()
    async def issue_refund(
        customer_id: int, order_id: int, reason: str
    ) -> dict[str, Any]:
        raise RuntimeError("database unavailable")

    return server, calls


async def run(model, server):
    graph = build_graph(model, server)
    return await graph.ainvoke(
        {"messages": [HumanMessage("question")], "customer_id": ALICE},
        {"recursion_limit": MAX_STEPS},
    )


async def test_tool_call_runs_and_result_goes_back_to_model(fake_mcp):
    model = ScriptedModel(
        [tool_call("get_order", {"order_id": 1}, "call_1"), AIMessage("done")]
    )
    result = await run(model, fake_mcp[0])
    tool_msg = result["messages"][2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_1"
    assert '"status": "delivered"' in tool_msg.content
    assert tool_msg in model.seen[1]
    assert result["messages"][-1].content == "done"


async def test_customer_id_comes_from_state_not_model(fake_mcp):
    model = ScriptedModel(
        [
            tool_call("get_order", {"order_id": BOB_ORDER, "customer_id": 2}, "c1"),
            AIMessage("done"),
        ]
    )
    await run(model, fake_mcp[0])
    assert fake_mcp[1] == [(ALICE, BOB_ORDER)]


async def test_every_tool_call_gets_one_tool_message_even_on_error(fake_mcp):
    reply = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_order", "args": {"order_id": 1}, "id": "ok"},
            {"name": "get_order", "args": {"order_id": "abc"}, "id": "bad_args"},
            {"name": "delete_order", "args": {}, "id": "unknown"},
        ],
    )
    result = await run(ScriptedModel([reply, AIMessage("done")]), fake_mcp[0])
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.status) for m in tool_msgs] == [
        ("ok", "success"),
        ("bad_args", "error"),
        ("unknown", "error"),
    ]


async def test_looping_model_hits_step_cap(fake_mcp):
    with pytest.raises(GraphRecursionError):
        await run(LoopingModel(), fake_mcp[0])
    assert len(fake_mcp[1]) < MAX_STEPS


async def test_server_side_tool_failure_becomes_error_tool_message(fake_mcp):
    model = ScriptedModel(
        [
            tool_call("issue_refund", {"order_id": 1, "reason": "broken"}, "r1"),
            AIMessage("done"),
        ]
    )
    result = await run(model, fake_mcp[0])
    tool_msg = result["messages"][2]
    assert tool_msg.status == "error"
    assert "issue_refund" in tool_msg.content
