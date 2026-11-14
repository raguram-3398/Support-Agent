import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langfuse import Evaluation

from support_agent.evals import (
    CHECKS,
    check_db_state,
    check_tools,
    gate,
    majority,
    missing,
    pass_counts,
    tool_calls,
    trial_results,
)
from support_agent.graph import TOOLS

TASKS = json.loads(Path("evals/tasks.json").read_text())
PASS = {"db_state": True, "tools": True, "judge": True}


def item(task_id, *passed):
    return SimpleNamespace(
        item=SimpleNamespace(metadata={"task_id": task_id}),
        evaluations=[
            Evaluation(name=check, value=int(ok))
            for check, ok in zip(CHECKS, passed, strict=True)
        ],
    )


def test_tool_calls_lists_names_in_order():
    messages = [
        HumanMessage("refund order 1"),
        AIMessage(
            "", tool_calls=[{"name": "get_order", "args": {"order_id": 1}, "id": "a"}]
        ),
        ToolMessage("{}", tool_call_id="a"),
        AIMessage(
            "",
            tool_calls=[
                {"name": "issue_refund", "args": {"order_id": 1}, "id": "b"},
                {"name": "get_order", "args": {"order_id": 2}, "id": "c"},
            ],
        ),
        AIMessage("Done."),
    ]
    assert tool_calls(messages) == ["get_order", "issue_refund", "get_order"]


def test_db_state_ignores_order_but_not_extra_refunds():
    assert check_db_state([2, 1], [1, 2])
    assert not check_db_state([1, 2], [1])
    assert not check_db_state([], [1])


def test_tools_needs_required_calls_and_no_forbidden_ones():
    assert check_tools(["list_orders", "get_order"], ["get_order"], ["issue_refund"])
    assert not check_tools(["list_orders"], ["get_order"], [])
    assert not check_tools(
        ["get_order", "issue_refund"], ["get_order"], ["issue_refund"]
    )


def test_trial_results_and_missing_find_unscored_checks():
    unscored = item("b", 1, 1, 1)
    unscored.evaluations = unscored.evaluations[:2]
    trial = trial_results([item("a", 1, 0, 1), unscored])
    assert trial["a"] == {"db_state": True, "tools": False, "judge": True}
    assert missing(trial, ["a", "b", "c"]) == [
        "b judge",
        "c db_state",
        "c tools",
        "c judge",
    ]


def test_majority_needs_two_of_three():
    trials = [
        {"a": {**PASS, "judge": False}},
        {"a": {**PASS, "judge": True}},
        {"a": {**PASS, "judge": False, "tools": False}},
    ]
    assert majority(trials) == {"a": {**PASS, "judge": False}}


def test_pass_counts():
    results = {"a": PASS, "b": {**PASS, "judge": False}}
    assert pass_counts(results) == {"n": 2, "db_state": 2, "tools": 2, "judge": 1}


def test_gate_passes_when_nothing_regresses():
    baseline = {"a": PASS, "b": {**PASS, "judge": False}}
    assert gate(baseline, baseline) == []
    assert gate({"a": PASS, "b": PASS}, baseline) == []


def test_gate_fails_on_a_swap_that_keeps_counts_equal():
    baseline = {"a": PASS, "b": {**PASS, "judge": False}}
    current = {"a": {**PASS, "judge": False}, "b": PASS}
    assert pass_counts(current) == pass_counts(baseline)
    assert gate(current, baseline) == ["a judge: passed in baseline, fails now"]


def test_gate_fails_when_a_baseline_task_is_absent():
    assert gate({}, {"a": PASS}) == [
        f"a {check}: passed in baseline, fails now" for check in CHECKS
    ]


def test_tasks_file_is_consistent():
    assert len({task["id"] for task in TASKS}) == len(TASKS) == 25
    for task in TASKS:
        expected = task["expected"]
        named = expected["must_call"] + expected["must_not_call"]
        assert set(named) <= set(TOOLS), task["id"]
        assert set(task["input"]["prior_refunds"]) <= set(expected["refunds"])
