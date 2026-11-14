import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from support_agent import graph as graph_module
from support_agent.graph import (
    AGENT_NODE_TIMEOUT_S,
    AGENT_TIMEOUT_REPLY,
    MAX_STEPS,
    MCP_TIMEOUT_S,
    MODEL_TIMEOUT_S,
    REJECTED_RESULT,
    TOOLS_NODE_TIMEOUT_S,
    TOOLS_TIMEOUT_RESULT,
    build_graph,
)

ALICE, BOB_ORDER = 1, 4
SMALL_ORDER, LARGE_ORDER, FAILING_ORDER = 1, 2, 3
SLOW_S = 1
TINY_TIMEOUT_S = 0.05


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


class SlowModel:
    async def ainvoke(self, messages):
        await asyncio.sleep(SLOW_S)
        return AIMessage("too late")


def tool_call(name, args, call_id):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def refund_call(order_id, call_id="r1"):
    return tool_call(
        "issue_refund", {"order_id": order_id, "reason": "broken"}, call_id
    )


THREAD = {"configurable": {"thread_id": "t1"}, "recursion_limit": MAX_STEPS}


async def run(model, server, checkpointer=None):
    graph = build_graph(model, checkpointer or InMemorySaver(), server)
    return await graph.ainvoke(
        {"messages": [HumanMessage("question")], "customer_id": ALICE}, THREAD
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
    model = ScriptedModel([refund_call(FAILING_ORDER), AIMessage("done")])
    result = await run(model, fake_mcp[0])
    tool_msg = result["messages"][2]
    assert tool_msg.status == "error"
    assert "issue_refund" in tool_msg.content


async def test_small_refund_runs_without_approval(fake_mcp):
    model = ScriptedModel([refund_call(SMALL_ORDER), AIMessage("refunded")])
    result = await run(model, fake_mcp[0])
    assert "__interrupt__" not in result
    assert fake_mcp[2] == [(ALICE, SMALL_ORDER)]


async def test_large_refund_pauses_then_runs_once_after_approval(fake_mcp):
    server, lookups, refunds = fake_mcp
    checkpointer = InMemorySaver()
    paused = await run(ScriptedModel([refund_call(LARGE_ORDER)]), server, checkpointer)
    assert paused["__interrupt__"][0].value == {
        "refunds": [{"order_id": LARGE_ORDER, "total_cents": 15000}]
    }
    assert refunds == []

    restarted = build_graph(
        ScriptedModel([AIMessage("refunded")]), checkpointer, server
    )
    result = await restarted.ainvoke(Command(resume=True), THREAD)
    assert refunds == [(ALICE, LARGE_ORDER)]
    assert lookups == [(ALICE, LARGE_ORDER)] * 2
    assert result["messages"][-1].content == "refunded"


async def test_rejected_refund_never_runs(fake_mcp):
    server, _, refunds = fake_mcp
    checkpointer = InMemorySaver()
    model = ScriptedModel([refund_call(LARGE_ORDER), AIMessage("not refunded")])
    await run(model, server, checkpointer)
    graph = build_graph(model, checkpointer, server)
    result = await graph.ainvoke(Command(resume=False), THREAD)
    assert refunds == []
    assert result["messages"][2].content == REJECTED_RESULT
    assert result["messages"][-1].content == "not refunded"


async def test_agent_timeout_ends_with_apology(fake_mcp, monkeypatch):
    monkeypatch.setattr(graph_module, "AGENT_NODE_TIMEOUT_S", TINY_TIMEOUT_S)
    result = await run(SlowModel(), fake_mcp[0])
    assert result["messages"][-1].content == AGENT_TIMEOUT_REPLY


async def test_tools_timeout_answers_every_call_and_returns_to_model(
    fake_mcp, monkeypatch
):
    monkeypatch.setattr(graph_module, "TOOLS_NODE_TIMEOUT_S", TINY_TIMEOUT_S)
    model = ScriptedModel([tool_call("list_orders", {}, "l1"), AIMessage("sorry")])
    result = await run(model, fake_mcp[0])
    tool_msg = result["messages"][2]
    assert (tool_msg.tool_call_id, tool_msg.status) == ("l1", "error")
    assert tool_msg.content == TOOLS_TIMEOUT_RESULT
    assert result["messages"][-1].content == "sorry"


def test_node_timeouts_fire_before_client_timeouts():
    assert TOOLS_NODE_TIMEOUT_S < MCP_TIMEOUT_S
    assert MODEL_TIMEOUT_S < AGENT_NODE_TIMEOUT_S
