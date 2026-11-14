import json
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from mcp import Client
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MODEL = "claude-haiku-5-5"
MODEL_TIMEOUT_S = 30
MODEL_MAX_RETRIES = 2
MAX_TOKENS = 1024
MAX_STEPS = 12
MCP_URL = "http://127.0.0.1:8000/mcp"
MCP_TIMEOUT_S = 10
AGENT_NODE_TIMEOUT_S = 45
TOOLS_NODE_TIMEOUT_S = 8
REFUND_APPROVAL_CENTS = 10_000

AGENT_TIMEOUT_REPLY = "Sorry, that took too long. Please try again."
TOOLS_TIMEOUT_RESULT = "Tool timed out. It may or may not have run."
REJECTED_RESULT = "Not run: a reviewer rejected the refund."

SYSTEM_PROMPT = """You are the customer support agent for a small online store.
You help the signed-in customer with their own orders. Use the tools to look up
orders; never guess order details. Amounts from tools are in cents; show them
to the customer in dollars.

Refund policy:
- Only delivered orders can be refunded.
- Placed or shipped orders cannot be refunded yet; tell the customer to ask
  again once the order is delivered.
- Cancelled orders were never charged, so there is nothing to refund.
- An order can be refunded once, for its full total. No partial refunds.
- Before issuing a refund, look up the order and confirm it qualifies, and
  have a reason from the customer.
- If an order does not qualify, explain which rule applies.

Keep replies short and plain."""


class GetOrder(BaseModel):
    """Look up one of the customer's orders by its order number."""

    model_config = ConfigDict(title="get_order")
    order_id: int = Field(description="The order number.")


class ListOrders(BaseModel):
    """List all of the customer's orders, newest first."""

    model_config = ConfigDict(title="list_orders")


class IssueRefund(BaseModel):
    """Refund the full total of one of the customer's delivered orders."""

    model_config = ConfigDict(title="issue_refund")
    order_id: int = Field(description="The order number to refund.")
    reason: str = Field(description="Why the customer wants the refund.")


TOOLS = {"get_order": GetOrder, "list_orders": ListOrders, "issue_refund": IssueRefund}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: int


def make_model():
    model = ChatAnthropic(
        model=MODEL,
        timeout=MODEL_TIMEOUT_S,
        max_retries=MODEL_MAX_RETRIES,
        max_tokens=MAX_TOKENS,
    )
    return model.bind_tools(list(TOOLS.values()))


def _reply_to_every_call(state: State, content: str) -> list[ToolMessage]:
    return [
        ToolMessage(
            content=content, tool_call_id=call["id"], name=call["name"], status="error"
        )
        for call in state["messages"][-1].tool_calls
    ]


def build_graph(model, checkpointer, mcp_server=MCP_URL):
    async def agent(state: State) -> dict:
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    async def approval(state: State) -> Command[Literal["tools", "agent"]]:
        refund_ids = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] == "issue_refund":
                try:
                    refund_ids.append(IssueRefund.model_validate(call["args"]).order_id)
                except ValidationError:
                    pass
        large = []
        if refund_ids:
            async with Client(mcp_server, read_timeout_seconds=MCP_TIMEOUT_S) as client:
                for order_id in refund_ids:
                    result = await client.call_tool(
                        "get_order",
                        {"customer_id": state["customer_id"], "order_id": order_id},
                    )
                    if result.is_error:
                        continue
                    order = result.structured_content["result"]
                    if order and order["total_cents"] > REFUND_APPROVAL_CENTS:
                        large.append(
                            {"order_id": order_id, "total_cents": order["total_cents"]}
                        )
        if not large or interrupt({"refunds": large}):
            return Command(goto="tools")
        return Command(
            goto="agent",
            update={"messages": _reply_to_every_call(state, REJECTED_RESULT)},
        )

    async def run_tools(state: State) -> dict:
        results = []
        async with Client(mcp_server, read_timeout_seconds=MCP_TIMEOUT_S) as client:
            for call in state["messages"][-1].tool_calls:
                try:
                    args = TOOLS[call["name"]].model_validate(call["args"]).model_dump()
                except (KeyError, ValidationError) as exc:
                    content, status = f"Invalid tool call: {exc}", "error"
                else:
                    args["customer_id"] = state["customer_id"]
                    result = await client.call_tool(call["name"], args)
                    if result.is_error:
                        content, status = result.content[0].text, "error"
                    else:
                        content = json.dumps(result.structured_content)
                        status = "success"
                results.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=call["id"],
                        name=call["name"],
                        status=status,
                    )
                )
        return {"messages": results}

    def agent_failed(state: State, error: NodeError) -> Command[Literal["__end__"]]:
        if not isinstance(error.error, NodeTimeoutError):
            raise error.error
        return Command(goto=END, update={"messages": [AIMessage(AGENT_TIMEOUT_REPLY)]})

    def tools_failed(state: State, error: NodeError) -> Command[Literal["agent"]]:
        if not isinstance(error.error, NodeTimeoutError):
            raise error.error
        return Command(
            goto="agent",
            update={"messages": _reply_to_every_call(state, TOOLS_TIMEOUT_RESULT)},
        )

    def route(state: State) -> str:
        return "approval" if state["messages"][-1].tool_calls else END

    graph = StateGraph(State)
    graph.add_node(
        "agent", agent, timeout=AGENT_NODE_TIMEOUT_S, error_handler=agent_failed
    )
    graph.add_node("approval", approval)
    graph.add_node(
        "tools", run_tools, timeout=TOOLS_NODE_TIMEOUT_S, error_handler=tools_failed
    )
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["approval", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
