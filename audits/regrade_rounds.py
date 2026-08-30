#!/usr/bin/env python3
"""Re-grade each optimisation round, and say whether the winning round changes.

AFlow and FlowBank pick the round to evaluate on held-out data *at test time*, from
the validation scores they stored during the search. Those scores were computed by
the scorer that was live at the time, so a scorer fix afterwards leaves the stored
ranking stale -- and the ranking is what decides which workflow gets reported.

Unlike the workflow *search* itself, this part is repairable without any GPU: every
round kept its per-item predictions, so the rounds can be re-graded and the choice
remade. This reports both rankings so the size of the difference is a number rather
than an assumption.

    python regrade_rounds.py --workspace third_party/aflow/workspace/SHARED_DROP \\
                             --dataset drop
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402


def regrade_csv(path: Path, dataset: str) -> tuple[float, int]:
    """(mean re-graded score, items) for one round's per-item file."""
    total = count = 0.0, 0
    accumulated = 0.0
    items = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            gold = (row.get("expected_output") or row.get("expected") or "").strip()
            reply = row.get("prediction") or row.get("output") or ""
            if not gold:
                continue
            # mmlu_pro needs an options row and mbpp needs the task's test
            # suite -- flat stubs crash both. collect._grading_row already
            # solves this (mbpp reconnects the gold code to its full row),
            # so reuse it rather than growing a second copy of the lesson.
            if dataset in ("mmlu_pro", "mbpp"):
                sys.path.insert(0, str(ROOT))
                from collect import _grading_row
                row = _grading_row(dataset, gold)
                if row is None:
                    continue
            else:
                row = {"ref_text": gold, "answer": gold, "code": gold}
            value, _ = bench.score(dataset, row, reply)
            accumulated += value
            items += 1
    return (accumulated / items if items else 0.0), items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    base = ROOT / args.workspace
    rounds = sorted((p for p in base.glob("workflows/round_*") if p.is_dir()),
                    key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not rounds:
        rounds = sorted((p for p in base.glob("round_*") if p.is_dir()),
                        key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not rounds:
        raise SystemExit(f"no round_* directories under {base}")

    print(f"  {'round':>6}{'stored':>10}{'re-graded':>12}{'delta':>9}{'items':>8}")
    table = []
    for round_dir in rounds:
        files = sorted(round_dir.glob("0.*.csv"))
        if not files:
            continue
        # Regrade EVERY validation csv in the round and keep the round's best.
        # "Newest file" is a trap: round dirs accumulate re-validations, and the
        # 2026-08-24 fd-storm wrote near-zero garbage csvs with the newest
        # mtimes -- for SHARED_MBPP round_9 that picked a 0.023 storm csv over
        # the round's healthy original and poisoned the winner selection.
        best = None
        for path in files:
            stored = float(re.match(r"0\.\d+", path.name).group(0))
            fresh, items = regrade_csv(path, args.dataset)
            if items and (best is None or fresh > best[1]):
                best = (stored, fresh, items, path.name)
        if best is None:
            continue
        stored, fresh, items, fname = best
        table.append((int(re.sub(r"\D", "", round_dir.name)), stored, fresh, items))
        print(f"  {table[-1][0]:>6}{stored:>10.4f}{fresh:>12.4f}"
              f"{(fresh - stored) * 100:>+8.2f}{items:>8}  {fname}")

    if not table:
        raise SystemExit("no per-item files in any round")
    best_stored = max(table, key=lambda r: r[1])
    best_fresh = max(table, key=lambda r: r[2])
    print(f"\n  best round by the stored scores   : round {best_stored[0]} "
          f"({best_stored[1]:.4f} stored, {best_stored[2]:.4f} re-graded)")
    print(f"  best round by the re-graded scores: round {best_fresh[0]} "
          f"({best_fresh[1]:.4f} stored, {best_fresh[2]:.4f} re-graded)")
    if best_stored[0] == best_fresh[0]:
        print("  -> the choice does not change: the scorer bug did not move the winner")
    else:
        gap = best_fresh[2] - best_stored[2]
        print(f"  -> the choice CHANGES. Evaluating the stored winner instead of the "
              f"re-graded one costs {gap * 100:.2f} validation points")


if __name__ == "__main__":
    main()
