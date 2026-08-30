#!/usr/bin/env python3
"""Feed every gold answer back in as if the model had said it. It must score 1.0.

The cleanest test of an extraction-and-comparison pipeline: if the reference
answer, written in the format the protocol asks for, does not score full marks
against itself, the pipeline is broken -- and no judgement about what the model
"meant" is involved.

It covers the two things that matter, in one pass:

  extraction  the wrapper is the shape our own ANSWER_FORMAT instruction asks
              for, so a failure means we cannot recover an answer from the format
              we ourselves demanded.
  comparison  the payload is the gold string verbatim, so a failure means
              normalisation mangles a value that is by definition correct.

Every failure is printed with the gold, the wrapper, and what the extractor
returned, because that triple is enough to see the cause.

    python test_gold_roundtrip.py                 # all datasets
    python test_gold_roundtrip.py --dataset drop --show 30
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402


def wrappers(dataset: str, gold: str) -> list[tuple[str, str]]:
    """(name, reply) pairs: the shapes a model legitimately answers in.

    The first of each list is the one our ANSWER_FORMAT instruction asks for. The
    others are forms the model uses anyway, and the tiered extractor exists
    precisely to accept them, so they are part of the contract too.
    """
    if dataset in ("math", "amc"):
        return [
            ("boxed", f"Some reasoning here.\n\\boxed{{{gold}}}"),
            ("boxed only", f"\\boxed{{{gold}}}"),
            ("answer is", f"Working it through, the answer is {gold}"),
            ("bold", f"Therefore the result is **{gold}**"),
            # The AFlow/FlowBank operator envelope. Graded as-is it scored 0 until
            # the scorer learned to strip it; frozen here so it cannot regress.
            ("xml envelope", f"<thought>working</thought>\n<answer>\\boxed{{{gold}}}</answer>"),
        ]
    if dataset == "drop":
        return [
            ("Answer: line", f"Some reasoning here.\n\nAnswer: {gold}"),
            ("answer is", f"Reading the passage, the answer is {gold}"),
            ("bare", gold),
            ("xml envelope", f"<thought>working</thought>\n<answer>{gold}</answer>"),
        ]
    if dataset == "mmlu_pro":
        return [
            ("Answer: (X)", f"Some reasoning here.\n\nAnswer: ({gold})"),
            # The shape that mis-graded 168 daao/mmlu_pro items (2026-08-24): the
            # first "Answer:" plus the newline let the unguarded capture take the
            # leading "A" of the SECOND word "Answer" instead of the letter.
            ("final-answer header", f"reasoning\n\n---\n\n### Final Answer:\nAnswer: ({gold})"),
            ("Answer: X", f"Some reasoning here.\n\nAnswer: {gold}"),
            ("boxed", f"Therefore \\boxed{{{gold}}}"),
            ("option word", f"So option {gold} is correct."),
            ("xml envelope", f"<thought>working</thought>\n<answer>{gold}</answer>"),
        ]
    if dataset == "mbpp":
        return [("code block", f"Here is the solution:\n```python\n{gold}\n```"),
                ("bare code", gold)]
    raise KeyError(dataset)


def gold_forms(dataset: str, row: dict) -> list[str]:
    """The gold strings to test. DROP ships several annotator answers."""
    raw = bench.gold(dataset, row)
    if dataset == "drop":
        return [a.strip() for a in str(raw).split("|") if a.strip()]
    if dataset == "mmlu_pro":
        # gold() returns the answer letter for MMLU-Pro.
        return [str(raw).strip()]
    return [str(raw)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, choices=list(bench.DATASETS))
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(bench.DATASETS)
    overall_failures = 0

    for dataset in datasets:
        rows = bench.load(dataset)
        if args.limit:
            rows = rows[: args.limit]
        tried = 0
        failures: list[tuple[str, str, str, float]] = []
        by_shape: collections.Counter = collections.Counter()

        for row in rows:
            for gold in gold_forms(dataset, row):
                for shape, reply in wrappers(dataset, gold):
                    tried += 1
                    try:
                        value, extracted = bench.score(dataset, row, reply)
                    except Exception as exc:  # noqa: BLE001
                        failures.append((shape, gold, f"RAISED {type(exc).__name__}", 0.0))
                        by_shape[shape] += 1
                        continue
                    if value < 0.999:
                        failures.append((shape, gold, extracted, value))
                        by_shape[shape] += 1

        rate = len(failures) / tried if tried else 0.0
        status = "OK" if not failures else "FAIL"
        print(f"\n  [{status}] {dataset}: {tried:,} gold answers replayed, "
              f"{len(failures):,} did not score 1.0 ({rate:.2%})")
        for shape, count in by_shape.most_common():
            print(f"        {shape:<14}{count:>7,} failure(s)")
        for shape, gold, extracted, value in failures[: args.show]:
            print(f"        [{shape}] gold={gold[:38]!r:<42} "
                  f"extracted={extracted[:38]!r:<42} score={value:.2f}")
        overall_failures += len(failures)

    print(f"\n  {overall_failures:,} failing case(s) in total")
    sys.exit(1 if overall_failures else 0)


if __name__ == "__main__":
    main()
