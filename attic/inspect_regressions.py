#!/usr/bin/env python3
"""Show the items a protocol change turned from right to wrong, with both replies.

A paired test says whether a difference is real; it cannot say whether the method
got worse at the task or merely stopped answering in a shape the grader reads. The
two look identical in the score column and call for opposite responses -- one is a
finding, the other is a bug in the adaptation -- so the predictions themselves have
to be read side by side.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, "shared")


def load(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("question") or "").strip()
            if question:
                rows[question] = row
    return rows


def tail(text: str, n: int = 180) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--show", type=int, default=6)
    parser.add_argument("--direction", choices=["worse", "better"], default="worse")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    changed = []
    for question in set(before) & set(after):
        try:
            b = float(before[question].get("score") or 0)
            a = float(after[question].get("score") or 0)
        except ValueError:
            continue
        if args.direction == "worse" and a < b - 1e-9:
            changed.append((a - b, question, before[question], after[question]))
        elif args.direction == "better" and a > b + 1e-9:
            changed.append((b - a, question, before[question], after[question]))
    changed.sort(key=lambda item: item[0])

    # Does the new prediction still contain the gold answer somewhere? If it does,
    # the answer was found and the grader did not see it -- an extraction problem,
    # not a reasoning one.
    text_present = 0
    for _delta, _q, b, a in changed:
        gold = (a.get("expected_output") or "").strip().strip("[]'\" ")
        if gold and gold.lower() in (a.get("prediction") or "").lower():
            text_present += 1
    print(f"  {len(changed)} item(s) went {args.direction}")
    print(f"  of those, the gold answer still appears somewhere in the new reply: "
          f"{text_present} ({text_present / max(len(changed), 1):.0%})")
    print("  -> a high share means the model found it and the grader did not read it\n")

    for delta, question, b, a in changed[: args.show]:
        print("=" * 96)
        print(f"  delta {delta:+.2f}   gold: {(a.get('expected_output') or '')[:70]!r}")
        flat_q = re.sub(r"\s+", " ", question)[:150]
        print(f"  Q: {flat_q}")
        print(f"  BEFORE (score {b.get('score')}): ...{tail(b.get('prediction'))}")
        print(f"  AFTER  (score {a.get('score')}): ...{tail(a.get('prediction'))}")


if __name__ == "__main__":
    main()
