#!/usr/bin/env python3
"""Frozen regression for the fifth scoring tier (order-free bare comma lists).

Source: the 2026-08-25 overnight incremental audit flagged 11 zero-scored math
records whose extracted span contains the gold. Manual adjudication: 10 are
genuinely wrong answers that must STAY zero; 1 is the same solution set
permuted and must now score 1. Every case is frozen here, plus guard cases
proving ordered tuples and intervals stay order-sensitive.

    envs/maas/bin/python audits/regression_comma_multiset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import bench  # noqa: E402

# (dataset, gold, model reply, expected score)
CASES = [
    # The miscarriage: identical solution set, permuted order.
    ("math", r"\frac{3}{4}, -\frac{3}{4}",
     r"The answer is \boxed{-\frac{3}{4}, \frac{3}{4}}", 1.0),
    # The ten genuinely-wrong siblings from the same audit batch.
    ("math", r"\sqrt{6}", r"The answer is \boxed{\frac{2\sqrt{6}}{3}}", 0.0),
    ("math", r"\sqrt{6}", r"The answer is \boxed{2\sqrt{6}}", 0.0),
    ("math", "715", r"The answer is \boxed{7150}", 0.0),
    ("math", "432", r"The answer is \boxed{432.25}", 0.0),
    ("math", r"\frac{3}{5}, \frac{117}{125}",
     r"The answer is \boxed{\frac{117}{125}, \frac{3}{5}, -\frac{3}{5}}", 0.0),
    ("math", "(0,1)", r"The answer is \boxed{[0, 1]}", 0.0),
    ("math", "3, 11, 33", r"The answer is \boxed{1, 3, 11, 33}", 0.0),
    ("math", "(-7,10)", r"The answer is \boxed{(7, -10)}", 0.0),
    # Guard cases: ordered things must never be treated as multisets.
    ("math", "(-7,10)", r"The answer is \boxed{(10, -7)}", 0.0),
    ("math", "(2,5)", r"The answer is \boxed{(5,2)}", 0.0),   # interval reversed
    ("math", "16:3", r"The answer is \boxed{3:16}", 0.0),      # ratio reversed
    # Multiset tier with sympy-equal elements, not just string-equal ones.
    ("amc", r"\frac{1}{2}, 3", r"The answer is \boxed{3, 0.5}", 1.0),
    # Identity (roundtrip shape) for a comma gold must of course still pass.
    ("math", r"\frac{3}{4}, -\frac{3}{4}",
     r"The answer is \boxed{\frac{3}{4}, -\frac{3}{4}}", 1.0),
    # Sixth tier (2026-08-25 whole-store audit): tuple gold vs column-vector
    # notation -- same ordered triple, must now score 1.
    ("math", "(7,21,35)",
     r"The answer is \boxed{\begin{pmatrix} 7 \\ 21 \\ 35 \end{pmatrix}}", 1.0),
    # Its genuinely-wrong siblings from the same batch must stay 0.
    ("math", r"\begin{pmatrix} 1 \\ 2 \\ -3 \end{pmatrix}",
     r"The answer is \boxed{\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}}", 0.0),
    ("math", r"\begin{pmatrix} 2 \\ -1 \\ -5 \end{pmatrix}",
     r"The answer is \boxed{\begin{pmatrix} 2 \\ 1 \\ 5 \end{pmatrix}}", 0.0),
    # Vectors are ORDERED: a permuted vector is wrong even with equal entries.
    ("math", "(7,21,35)",
     r"The answer is \boxed{\begin{pmatrix} 7 \\ 35 \\ 21 \end{pmatrix}}", 0.0),
    # vmatrix is a determinant, not a vector: never equated with a tuple.
    ("math", "(1,2)",
     r"The answer is \boxed{\begin{vmatrix} 1 \\ 2 \end{vmatrix}}", 0.0),
]


def main() -> None:
    failures = 0
    for dataset, gold, reply, want in CASES:
        row = {"answer": gold, "solution": ""}
        got, extracted = bench.score(dataset, row, reply)
        status = "ok " if got == want else "FAIL"
        if got != want:
            failures += 1
        print(f"[{status}] {dataset} gold={gold!r:42} want={want} got={got}"
              f"  extracted={extracted!r}")
    if failures:
        print(f"\n{failures} case(s) FAILED")
        sys.exit(1)
    print(f"\nall {len(CASES)} cases pass")


if __name__ == "__main__":
    main()
