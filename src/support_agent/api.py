import asyncio
import os
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

from support_agent.auth import JWKS_URL, Verifier
from support_agent.graph import MAX_STEPS, build_graph, make_model

API_AUDIENCE = "support-api"
CUSTOMER_CLAIM = "customer_id"

verifier = Verifier(JWKS_URL, API_AUDIENCE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.environ["CHECKPOINT_DB_URL"]
    async with AsyncPostgresSaver.from_conn_string(url) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(make_model(), checkpointer)
        yield
    get_client().flush()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool


def _unauthorized(challenge: str) -> HTTPException:
    return HTTPException(
        401, "Invalid or missing token.", {"WWW-Authenticate": challenge}
    )


async def current_customer(request: Request) -> int:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized("Bearer")
    try:
        claims = await asyncio.to_thread(verifier.verify, token)
        return int(claims[CUSTOMER_CLAIM])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        raise _unauthorized('Bearer error="invalid_token"') from None


CustomerId = Annotated[int, Depends(current_customer)]


async def _is_paused(request: Request, thread_id: str, customer_id: int) -> bool:
    config = {"configurable": {"thread_id": thread_id}}
    state = await request.app.state.graph.aget_state(config)
    if state.values and state.values["customer_id"] != customer_id:
        raise HTTPException(404, "Thread not found.")
    return bool(state.interrupts)


async def not_paused(
    request: Request, body: ChatRequest, customer_id: CustomerId
) -> int:
    if await _is_paused(request, body.thread_id, customer_id):
        raise HTTPException(409, "Thread is waiting for approval.")
    return customer_id


async def paused(
    request: Request, body: ApproveRequest, customer_id: CustomerId
) -> int:
    if not await _is_paused(request, body.thread_id, customer_id):
        raise HTTPException(409, "Thread is not waiting for approval.")
    return customer_id


def _summary(message) -> dict:
    if isinstance(message, AIMessage):
        return {"text": message.text, "tool_calls": message.tool_calls}
    if isinstance(message, ToolMessage):
        return {"tool": message.name, "status": message.status, "result": message.text}
    return {"text": message.text}


async def _stream(graph, graph_input, thread_id: str) -> AsyncIterable[ServerSentEvent]:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_STEPS,
        "callbacks": [CallbackHandler()],
    }
    try:
        async for update in graph.astream(graph_input, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":
                    yield ServerSentEvent(event="approval_required", data=data[0].value)
                elif data and data.get("messages"):
                    messages = [_summary(m) for m in data["messages"]]
                    yield ServerSentEvent(event=node, data=messages)
    except GraphRecursionError:
        yield ServerSentEvent(event="error", data=f"Step cap of {MAX_STEPS} reached.")
    yield ServerSentEvent(event="done", data="")


@app.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    customer_id: Annotated[int, Depends(not_paused)],
) -> AsyncIterable[ServerSentEvent]:
    graph_input = {"messages": [HumanMessage(body.message)], "customer_id": customer_id}
    async for event in _stream(request.app.state.graph, graph_input, body.thread_id):
        yield event


@app.post(
    "/approve", response_class=EventSourceResponse, dependencies=[Depends(paused)]
)
async def approve(
    request: Request, body: ApproveRequest
) -> AsyncIterable[ServerSentEvent]:
    graph_input = Command(resume=body.approved)
    async for event in _stream(request.app.state.graph, graph_input, body.thread_id):
        yield event
