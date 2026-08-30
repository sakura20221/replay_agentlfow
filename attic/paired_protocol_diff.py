#!/usr/bin/env python3
"""Is a score difference between two protocols larger than the run-to-run noise?

Two runs of the same method on the same 1,000 items are not two independent
samples: the items are shared, so most of the variance is "this question is hard",
which cancels when the comparison is done per item. An unpaired confidence
interval on each score separately therefore overstates the uncertainty badly, and
is the wrong instrument for "did this change anything".

So the two per-item CSVs are joined on the question text and the difference is
tested paired: a bootstrap over items, plus a count of how many items moved in
each direction. DROP is scored by F1, so the per-item value is continuous and the
paired bootstrap is the right test rather than McNemar.

    python paired_protocol_diff.py --before <csv> --after <csv> --label daao/drop
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys


def load(path: str) -> dict[str, float]:
    """question -> score. Later rows win, matching how the runners append."""
    rows: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            try:
                rows[question] = float(row.get("score") or 0.0)
            except ValueError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    shared = sorted(set(before) & set(after))
    if not shared:
        raise SystemExit("no questions in common; the two files are not comparable")

    diffs = [after[q] - before[q] for q in shared]
    mean_before = statistics.mean(before[q] for q in shared)
    mean_after = statistics.mean(after[q] for q in shared)
    delta = mean_after - mean_before

    rng = random.Random(20260823)
    boot = []
    n = len(diffs)
    for _ in range(args.resamples):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    low, high = boot[int(args.resamples * 0.025)], boot[int(args.resamples * 0.975)]
    # Two-sided p by the bootstrap's own sign: the share of resamples on the other
    # side of zero, doubled.
    share_le0 = sum(1 for b in boot if b <= 0) / args.resamples
    p = 2 * min(share_le0, 1 - share_le0)

    worse = sum(1 for d in diffs if d < -1e-9)
    better = sum(1 for d in diffs if d > 1e-9)
    same = n - worse - better

    print(f"  {args.label}   {n:,} questions in common "
          f"({len(before):,} before, {len(after):,} after)")
    print(f"  before {mean_before:.4f}   after {mean_after:.4f}   "
          f"delta {delta:+.4f} ({delta * 100:+.2f} points)")
    print(f"  paired bootstrap 95% CI on the delta: [{low * 100:+.2f}, {high * 100:+.2f}] points")
    print(f"  two-sided p = {p:.4f}")
    print(f"  items: {better:,} improved, {worse:,} got worse, {same:,} unchanged")
    if low <= 0 <= high:
        print("  -> the interval spans zero: not distinguishable from run-to-run noise")
    else:
        print("  -> the interval excludes zero: a real difference, not noise")
    # Where the change is concentrated: a handful of items swinging fully is a
    # different story from a broad small drift.
    big = sorted(diffs)[:5]
    print(f"  five largest per-item losses: {[round(x, 2) for x in big]}")
    print(f"  share of the total delta from items that changed by more than 0.5: "
          f"{sum(d for d in diffs if abs(d) > 0.5) / (delta * n):.0%}"
          if delta else "")


if __name__ == "__main__":
    main()
