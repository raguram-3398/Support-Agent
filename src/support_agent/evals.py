import os
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import Evaluation
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.graph import (
    MAX_STEPS,
    MODEL,
    MODEL_MAX_RETRIES,
    MODEL_TIMEOUT_S,
    build_graph,
    make_model,
)

CHECKS = ("db_state", "tools", "judge")
TRIALS = 3
MAJORITY = 2
JUDGE_MAX_TOKENS = 512

JUDGE_PROMPT = """You grade one reply from a store's customer support agent.
You see the customer's message, the agent's final reply, and one criterion.
Decide only whether the reply meets the criterion. Do not reward length,
politeness or extra detail. Write a one- or two-sentence critique, then the
verdict.

Customer message:
{question}

Agent reply:
{reply}

Criterion:
{criterion}"""


class Verdict(BaseModel):
    critique: str = Field(description="One or two sentences on why.")
    passed: bool = Field(description="True only if the reply meets the criterion.")


def tool_calls(messages) -> list[str]:
    return [
        call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def check_db_state(refunded: list[int], expected: list[int]) -> bool:
    return sorted(refunded) == sorted(expected)


def check_tools(called: list[str], must_call: list[str], must_not: list[str]) -> bool:
    return set(must_call) <= set(called) and not set(must_not) & set(called)


def trial_results(item_results) -> dict[str, dict[str, bool]]:
    return {
        result.item.metadata["task_id"]: {
            e.name: e.value == 1 for e in result.evaluations if e.name in CHECKS
        }
        for result in item_results
    }


def missing(trial: dict[str, dict[str, bool]], task_ids: list[str]) -> list[str]:
    return [
        f"{task_id} {check}"
        for task_id in task_ids
        for check in CHECKS
        if check not in trial.get(task_id, {})
    ]


def majority(trials: list[dict[str, dict[str, bool]]]) -> dict[str, dict[str, bool]]:
    return {
        task_id: {
            check: sum(trial[task_id][check] for trial in trials) >= MAJORITY
            for check in CHECKS
        }
        for task_id in trials[0]
    }


def pass_counts(results: dict[str, dict[str, bool]]) -> dict[str, int]:
    counts = {check: sum(r[check] for r in results.values()) for check in CHECKS}
    return {"n": len(results), **counts}


def gate(
    current: dict[str, dict[str, bool]], baseline: dict[str, dict[str, bool]]
) -> list[str]:
    return [
        f"{task_id} {check}: passed in baseline, fails now"
        for task_id, checks in baseline.items()
        for check, passed in checks.items()
        if passed and not current.get(task_id, {}).get(check, False)
    ]


async def _reset_refunds(prior_refunds: list[int]) -> None:
    engine = create_async_engine(os.environ["DATABASE_URL_ADMIN"])
    try:
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE refunds"))
            for order_id in prior_refunds:
                await conn.execute(
                    text(
                        "INSERT INTO refunds (order_id, amount_cents, reason) "
                        "SELECT id, total_cents, 'earlier refund' "
                        "FROM orders WHERE id = :id"
                    ),
                    {"id": order_id},
                )
    finally:
        await engine.dispose()


async def _refunded_order_ids() -> list[int]:
    engine = create_async_engine(os.environ["DATABASE_URL_ADMIN"])
    try:
        async with engine.connect() as conn:
            rows = await conn.execute(text("SELECT order_id FROM refunds"))
            return [row.order_id for row in rows]
    finally:
        await engine.dispose()


async def run_task(*, item, **kwargs) -> dict:
    task = item.input
    await _reset_refunds(task["prior_refunds"])
    graph = build_graph(make_model(), InMemorySaver())
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": MAX_STEPS,
        "callbacks": [CallbackHandler()],
    }
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(task["question"])],
            "customer_id": task["customer_id"],
        },
        config,
    )
    while "__interrupt__" in result and task["approve"] is not None:
        result = await graph.ainvoke(Command(resume=task["approve"]), config)
    return {
        "reply": result["messages"][-1].text,
        "tools": tool_calls(result["messages"]),
        "refunded": await _refunded_order_ids(),
        "paused": "__interrupt__" in result,
    }


def db_state(*, output, expected_output, **kwargs) -> Evaluation:
    passed = check_db_state(output["refunded"], expected_output["refunds"])
    return Evaluation(
        name="db_state",
        value=int(passed),
        comment=f"refunded {output['refunded']}, expected {expected_output['refunds']}",
    )


def tools(*, output, expected_output, **kwargs) -> Evaluation:
    passed = check_tools(
        output["tools"], expected_output["must_call"], expected_output["must_not_call"]
    )
    return Evaluation(name="tools", value=int(passed), comment=str(output["tools"]))


def make_judge():
    model = ChatAnthropic(
        model=MODEL,
        timeout=MODEL_TIMEOUT_S,
        max_retries=MODEL_MAX_RETRIES,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    return model.with_structured_output(Verdict)


async def judge(*, input, output, expected_output, **kwargs) -> Evaluation:
    verdict = await make_judge().ainvoke(
        JUDGE_PROMPT.format(
            question=input["question"],
            reply=output["reply"],
            criterion=expected_output["criterion"],
        )
    )
    return Evaluation(name="judge", value=int(verdict.passed), comment=verdict.critique)
