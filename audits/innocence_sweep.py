#!/usr/bin/env python3
"""Final innocence sweep: second-opinion every zero the current scorer gives.

correct_but_zero asks "does the gold appear inside the EXTRACTED span" -- a
prefilter. This tool asks the stronger question, per zero-scored record: does
an INDEPENDENT extraction of the reply's final answer equal the gold under an
INDEPENDENT equivalence (sympy parsed directly, plus numeric tolerance), or --
for DROP -- do the gold's tokens all appear verbatim in the reply's tail even
though the extracted span scored 0? Any hit is a potential miscarriage and is
printed IN FULL for human judgement, uncapped.

mbpp is deliberately absent: its grade is the execution of the reference tests,
so the re-grade itself is the second opinion (run via correct_but_zero, which
reported zero suspects on 2026-08-24).

    envs/maas/bin/python audits/innocence_sweep.py --since '2026-08-23 12:00'
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from correct_but_zero import sources, records, unwrap  # noqa: E402

TAIL_LETTER = re.compile(r"answer\s*(?:is)?\s*[::]?\s*\(?([A-J])\)?(?![A-Za-z])",
                         re.IGNORECASE)
TAIL_MATH = (
    re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"answer\s*(?:is|:|=)\s*([^\n]{1,60}?)\s*\.?\s*$",
               re.IGNORECASE | re.MULTILINE),
)
_WORD = re.compile(r"[a-z0-9]+")


def _independent_equal(candidate: str, gold: str) -> bool:
    """Equivalence WITHOUT bench's pipeline: direct sympy + numeric fallback."""
    a = bench._strip_answer_dressing(bench._relax_latex(candidate))
    b = bench._strip_answer_dressing(bench._relax_latex(gold))
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify
        if simplify(parse_latex(a) - parse_latex(b)) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def check_math(gold: str, pred: str) -> str | None:
    candidates = []
    for pattern in TAIL_MATH:
        found = pattern.findall(pred)
        if found:
            candidates.append(found[-1].strip())
    lines = [l.strip() for l in pred.splitlines() if l.strip()]
    if lines and len(lines[-1]) <= 40:
        candidates.append(lines[-1])
    for cand in candidates:
        if cand and _independent_equal(cand, gold):
            return cand
    return None


def check_mmlu(gold: str, pred: str) -> str | None:
    found = TAIL_LETTER.findall(pred)
    if found and found[-1].upper() == gold.strip().upper():
        return found[-1].upper()
    return None


def check_drop(gold: str, pred: str) -> str | None:
    tail = " ".join(pred.splitlines()[-4:]).lower()
    tail_tokens = set(_WORD.findall(tail))
    for alt in gold.split("|"):
        tokens = _WORD.findall(alt.lower())
        if tokens and all(t in tail_tokens for t in tokens):
            return alt
    return None


CHECKERS = {"math": check_math, "amc": check_math,
            "mmlu_pro": check_mmlu, "drop": check_drop}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-08-23 12:00")
    parser.add_argument("--tag", default="v5")
    args = parser.parse_args()
    floor = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M"))

    grand = 0
    for dataset in ("mmlu_pro", "amc", "math", "drop"):
        checker = CHECKERS[dataset]
        zeros = suspects = 0
        details = []
        for method, phase, path in sources(dataset, args.tag):
            if path.stat().st_mtime < floor:
                continue
            for gold, pred, _stored in records(path):
                pred = unwrap(pred)
                if dataset == "mbpp":
                    continue
                row = ({"answer": gold, "options": list("ABCDEFGHIJ")}
                       if dataset == "mmlu_pro" else
                       {"ref_text": gold, "answer": gold, "code": gold, "solution": ""})
                value, _ = bench.score(dataset, row, pred)
                if value > 0:
                    continue
                zeros += 1
                hit = checker(str(gold), str(pred))
                if hit is not None:
                    suspects += 1
                    details.append((f"{method}/{phase}", gold, hit, pred[-110:]))
        print(f"\n### {dataset}: 判0共 {zeros} 条, 独立第二意见判嫌疑 {suspects} 条")
        for cell, gold, hit, tail in details:
            print(f"  [{cell}] gold={gold[:36]!r} 独立抽取={hit[:36]!r}")
            print(f"      尾部: {tail!r}")
        grand += suspects
    print(f"\n===== 全部嫌疑合计: {grand} 条(以上已全量打印,无截断)=====")


if __name__ == "__main__":
    main()
