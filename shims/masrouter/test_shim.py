#!/usr/bin/env python3
"""Verify the MasRouter shim before any GPU time is spent.

    cd third_party/masrouter && python ../../shims/masrouter/test_shim.py

Beyond grading, this checks the task label per dataset. MasRouter trains a
classifier over its three task types with a cross-entropy loss, so a wrong or
constant label teaches the router that every query is the same kind of problem --
a failure that produces plausible-looking accuracy numbers while the routing
decision it is supposed to learn never gets a useful gradient.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from Datasets.shared_dataset import (  # noqa: E402
    TASK_LABEL,
    load_shared_dataset,
    shared_score,
    shared_task_labels,
)

EXPECTED_LABEL = {"math": 0, "amc": 0, "mbpp": 2, "drop": 1, "mmlu_pro": 1}
LABEL_NAME = {0: "Math", 1: "Commonsense", 2: "Code"}

WRONG = {
    "math": "the answer is -999999",
    "amc": "the answer is -999999",
    "mbpp": "def nope():\n    return None\n",
    "drop": "Answer: zzzz nonexistent",
    "mmlu_pro": "I decline to answer.",
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


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


for name in EXPECTED_LABEL:
    print(f"\n=== {name} ===")
    check(f"{name}: task label is {LABEL_NAME[EXPECTED_LABEL[name]]}",
          TASK_LABEL[name] == EXPECTED_LABEL[name], f"got {TASK_LABEL[name]}")

    test_records = load_shared_dataset(name, split="test")
    train_records = load_shared_dataset(name, split="train")
    check(f"{name}: test split loads", len(test_records) > 0, f"n={len(test_records)}")
    check(f"{name}: search split loads", len(train_records) > 0, f"n={len(train_records)}")
    check(f"{name}: record has problem+row",
          all({"problem", "row"} <= set(r) for r in test_records[:5]),
          str(sorted(test_records[0]))[:50])
    check(f"{name}: query text non-empty", bool(test_records[0]["problem"].strip()),
          f"{len(test_records[0]['problem'])} chars")

    labels = shared_task_labels(name, test_records[:4])
    check(f"{name}: batch labels uniform and correct",
          labels == [EXPECTED_LABEL[name]] * 4, str(labels))

    sample = [test_records[i] for i in (0, len(test_records) // 2, len(test_records) - 1)]
    good = [shared_score(name, r, gold_reply(name, r["row"])) for r in sample]
    check(f"{name}: gold answer scores 1.0", all(v >= 0.99 for v in good),
          str([round(v, 3) for v in good]))
    bad = [shared_score(name, r, WRONG[name]) for r in sample]
    check(f"{name}: wrong answer scores 0", all(v < 0.5 for v in bad),
          str([round(v, 3) for v in bad]))

print("\n=== derived runner ===")
runner = Path("Experiments") / "run_shared.py"
check("run_shared.py exists", runner.exists())
if runner.exists():
    try:
        py_compile.compile(str(runner), doraise=True)
        check("run_shared.py compiles", True)
    except py_compile.PyCompileError as exc:
        check("run_shared.py compiles", False, str(exc)[:200])
    text = runner.read_text(encoding="utf-8")
    check("no leftover MATH_get_predict", "MATH_get_predict" not in text)
    check("grading in both loops", text.count("shared_score(args.shared_dataset") == 2,
          str(text.count("shared_score(args.shared_dataset")))
    check("task labels in both loops", text.count("shared_task_labels(") == 2,
          str(text.count("shared_task_labels(")))

print("\n" + "=" * 58)
if failures:
    print(f"MasRouter SHIM TEST FAILED ({len(failures)})")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("MasRouter SHIM TEST PASSED")
