#!/usr/bin/env python3
"""Re-grade a finished run from the predictions it already stored.

A scorer bug does not require re-running anything: the model's replies are on
disk, and grading is a pure function of (reply, gold). This recomputes the score
for a MaAS-family per-item CSV with the current scorer, and reports the score the
file was written with alongside it, so the size of the grading error is visible
rather than asserted.

    python rescore_csv.py --csv <path> --dataset drop
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, "shared")
import bench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset", default="drop")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    # The gold comes from the CSV row itself, not from a lookup into the split.
    #
    # An earlier version matched each prediction back to a dataset row by the first
    # 160 characters of the question and scored against that row. The keys collided
    # -- DROP items share long passages -- so predictions were graded against other
    # questions' answers, and the harness reported a 33-point collapse that was
    # entirely its own bug. The runners already store the gold they used, in the
    # same pipe-joined form bench.gold() returns, so no lookup is needed.
    stored_total = new_total = matched = unmatched = 0
    changed = []
    with open(args.csv, newline="", encoding="utf-8", errors="replace") as handle:
        for record in csv.DictReader(handle):
            expected = (record.get("expected_output") or "").strip()
            if not expected:
                unmatched += 1
                continue
            matched += 1
            try:
                stored = float(record.get("score") or 0.0)
            except ValueError:
                stored = 0.0
            row = {"ref_text": expected, "answer": expected, "code": expected}
            fresh, extracted = bench.score(args.dataset, row,
                                           record.get("prediction") or "")
            stored_total += stored
            new_total += fresh
            if abs(fresh - stored) > 1e-9:
                changed.append((fresh - stored, expected, extracted))

    if not matched:
        raise SystemExit(f"no rows matched the {args.dataset} split; wrong dataset?")
    print(f"  {args.label or args.csv}")
    print(f"  {matched:,} item(s) matched, {unmatched} unmatched")
    print(f"  score as written : {stored_total / matched:.4f}")
    print(f"  score re-graded  : {new_total / matched:.4f}   "
          f"({(new_total - stored_total) / matched * 100:+.2f} points)")
    print(f"  items whose grade changed: {len(changed)}")
    for delta, gold, extracted in sorted(changed, key=lambda x: -x[0])[:4]:
        print(f"    {delta:+.2f}  gold={gold!r:<20} now extracts {extracted!r}")


if __name__ == "__main__":
    main()
