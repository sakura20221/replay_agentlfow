#!/usr/bin/env python3
"""Verify the AFlow shim before any GPU time is spent.

    cd third_party/aflow && python ../../shims/aflow/test_shim.py

The DROP case is the one worth watching: AFlow's own evaluate_problem maximises
F1 over "|"-separated fragments of the reply, while the shared layer extracts the
stated answer first. The override has to be in effect, otherwise DROP is graded
one way for AFlow and another way for the other six methods.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from benchmarks.shared_benchmarks import SHARED_DATASET_CONFIGS, shared_bench  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


WRONG = {
    "math": "the answer is -999999",
    "amc": "the answer is -999999",
    "mbpp": "def nope():\n    return None\n",
    "drop": "Answer: zzzz nonexistent",
    "mmlu_pro": "I decline to answer.",
}


def gold_reply(name: str, row: dict) -> str:
    if name in ("math", "amc"):
        return f"\\boxed{{{row['answer']}}}"
    if name == "mbpp":
        return row["code"]
    if name == "drop":
        return f"Answer: {row['answers'][0]}"
    if name == "mmlu_pro":
        return f"Answer: ({row['answer']})"
    raise KeyError(name)


class FakeGraph:
    def __init__(self, reply: str):
        self.reply = reply

    async def __call__(self, prompt, *extra):
        return self.reply, 0.0


print("=== registered datasets ===")
check("5 datasets registered", len(SHARED_DATASET_CONFIGS) == 5, str(sorted(SHARED_DATASET_CONFIGS)))


async def run() -> None:
    for key, cls in sorted(SHARED_DATASET_CONFIGS.items()):
        name = cls.SHARED_DATASET
        row = shared_bench.load(name)[0]
        benchmark = cls(name=key, file_path="unused", log_path="/tmp")

        result = await benchmark.evaluate_problem(dict(row), FakeGraph(gold_reply(name, row)))
        check(f"{key}: tuple arity 5", len(result) == 5, f"got {len(result)}")
        check(f"{key}: gold answer scores 1.0", result[3] >= 0.99, f"score={result[3]}")

        bad = await benchmark.evaluate_problem(dict(row), FakeGraph(WRONG[name]))
        check(f"{key}: wrong answer scores 0", bad[3] < 0.5, f"score={bad[3]}")

        columns = benchmark.get_result_columns()
        check(f"{key}: result columns match AFlow's contract",
              columns == ["inputs", "prediction", "expected_output", "score", "cost"], str(columns))


asyncio.run(run())

print("\n=== data files AFlow will actually read ===")
for key, cls in sorted(SHARED_DATASET_CONFIGS.items()):
    for suffix in ("test", "validate"):
        path = Path("data") / "datasets" / f"{key.lower()}_{suffix}.jsonl"
        ok = path.exists() and path.stat().st_size > 0
        detail = ""
        if ok:
            with path.open(encoding="utf-8") as handle:
                rows = sum(1 for line in handle if line.strip())
            detail = f"{rows} rows"
        check(f"{path.name}", ok, detail)

print("\n=== DROP grading really goes through the shared layer ===")
drop_cls = SHARED_DATASET_CONFIGS["SHARED_DROP"]
drop_row = shared_bench.load("drop")[0]
verbose = ("Let me work through the passage carefully. " * 30) + f"Answer: {drop_row['answers'][0]}"
benchmark = drop_cls(name="SHARED_DROP", file_path="unused", log_path="/tmp")
score, _ = shared_bench.score("drop", drop_row, verbose)
# AFlow's native DROP path would score this near zero: it never extracts the
# stated answer, so a long reply destroys token-level precision.
check("verbose reply still scores high via extraction", score >= 0.9, f"score={score}")

print("\n" + "=" * 58)
if failures:
    print(f"AFlow SHIM TEST FAILED ({len(failures)})")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("AFlow SHIM TEST PASSED")
