import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from langfuse import get_client

from support_agent.evals import (
    CHECKS,
    TRIALS,
    db_state,
    gate,
    judge,
    majority,
    missing,
    pass_counts,
    run_task,
    tools,
    trial_results,
)

DATASET = "support-agent-tasks"
EXPERIMENT = "support-agent-gate"
TASKS = Path("evals/tasks.json")
BASELINE = Path("evals/baseline.json")


def upload(langfuse, tasks: list[dict]) -> None:
    langfuse.create_dataset(name=DATASET)
    for task in tasks:
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            id=f"{DATASET}-{task['id']}",
            input=task["input"],
            expected_output=task["expected"],
            metadata={"task_id": task["id"]},
        )


def print_failures(trial_number: int, item_results) -> None:
    for item_result in item_results:
        task_id = item_result.item.metadata["task_id"]
        scored = {e.name: e for e in item_result.evaluations}
        failed = [c for c in CHECKS if c in scored and scored[c].value != 1]
        for check in failed:
            print(f"FAIL t{trial_number} {task_id} {check}: {scored[check].comment}")
        if failed:
            print(f"     reply: {item_result.output['reply']!r}")


def main(save_baseline: bool) -> int:
    tasks = json.loads(TASKS.read_text())
    task_ids = [task["id"] for task in tasks]
    langfuse = get_client()
    upload(langfuse, tasks)
    dataset = langfuse.get_dataset(DATASET)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    trials = []
    for trial_number in range(1, TRIALS + 1):
        result = dataset.run_experiment(
            name=EXPERIMENT,
            run_name=f"{EXPERIMENT} {stamp} t{trial_number}",
            task=run_task,
            evaluators=[db_state, tools, judge],
            max_concurrency=1,
        )
        print_failures(trial_number, result.item_results)
        trial = trial_results(result.item_results)
        gaps = missing(trial, task_ids)
        for gap in gaps:
            print(f"NOT SCORED t{trial_number} {gap}")
        if gaps:
            return 1
        trials.append(trial)

    results = majority(trials)
    print(json.dumps(pass_counts(results)))

    if save_baseline:
        BASELINE.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Baseline saved to {BASELINE}.")
        return 0

    failures = gate(results, json.loads(BASELINE.read_text()))
    for failure in failures:
        print(f"GATE FAIL {failure}")
    print("GATE FAIL" if failures else "GATE PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    sys.exit(main(parser.parse_args().save_baseline))
