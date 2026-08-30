#!/usr/bin/env python3
"""Prove the shared scorers actually grade, before any GPU time is spent.

A scorer that silently returns 0.0 for everything looks exactly like a method
that performs badly, so each dataset is checked in both directions: feeding the
gold answer must score 1, and feeding a wrong answer must score 0.
"""

from __future__ import annotations

import sys

import bench

WRONG = {
    "math": "The answer is -999999",
    "amc": "The answer is -999999",
    "drop": "zzzzz nonexistent answer zzzzz",
    "mmlu_pro": "I am not going to pick an option.",
    "mbpp": "def totally_wrong_function():\n    return None\n",
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def gold_prediction(name: str, row: dict) -> str:
    """Construct a prediction that should be graded correct."""
    if name in ("math", "amc"):
        return f"After working through it, the answer is \\boxed{{{row['answer']}}}"
    if name == "mbpp":
        return row["code"]
    if name == "drop":
        return row["answers"][0] if row.get("answers") else row["ref_text"].split("|")[0]
    if name == "mmlu_pro":
        return f"Reasoning omitted. The answer is ({row['answer']})"
    raise KeyError(name)


for dataset in bench.DATASETS:
    rows = bench.load(dataset)
    print(f"\n=== {dataset}  n={len(rows)}  metric={bench.metric_name(dataset)} ===")

    # Several rows, so one unusual item cannot hide a broken scorer.
    sample = [rows[i] for i in (0, len(rows) // 3, 2 * len(rows) // 3, len(rows) - 1)]

    positives = []
    for row in sample:
        value, extracted = bench.score(dataset, row, gold_prediction(dataset, row))
        positives.append(value)
    hit = sum(1 for v in positives if v >= 0.99)
    check(f"{dataset}: gold answer graded correct", hit == len(positives),
          f"{hit}/{len(positives)} scored 1.0 (values={[round(v, 3) for v in positives]})")

    negatives = [bench.score(dataset, row, WRONG[dataset])[0] for row in sample]
    check(f"{dataset}: wrong answer graded incorrect", all(v < 0.5 for v in negatives),
          f"values={[round(v, 3) for v in negatives]}")

    text = bench.question_text(dataset, rows[0])
    check(f"{dataset}: question_text non-empty", bool(text.strip()), f"{len(text)} chars")


# MBPP's test_imports field is not limited to imports: one official test item
# constructs object fixtures there. Replaying that gold solution catches a
# harness that executes the asserts without first executing their setup.
mbpp_setup_rows = [row for row in bench.load("mbpp") if row.get("test_imports")]
mbpp_setup_values = [
    bench.score("mbpp", row, gold_prediction("mbpp", row))[0]
    for row in mbpp_setup_rows
]
check(
    "mbpp: test_imports setup executed",
    bool(mbpp_setup_rows) and all(value >= 0.99 for value in mbpp_setup_values),
    f"{sum(value >= 0.99 for value in mbpp_setup_values)}/{len(mbpp_setup_rows)} scored 1.0",
)

print("\n" + "=" * 62)
if failures:
    print(f"SHARED BENCH TEST FAILED ({len(failures)})")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("SHARED BENCH TEST PASSED - one grading standard is live")
