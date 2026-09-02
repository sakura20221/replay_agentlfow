#!/usr/bin/env python3
"""Per (method, dataset), does any prompt actually sent carry another task's identity?

Install-time checks assert what a file contains. They cannot catch a layer that
re-reads the unpatched file at runtime -- which is exactly what MasRouter's
RoleRegistry does, loading `MAR/Roles/{domain}/{role}.json` fresh when each agent
is built, so an adapted copy in a sibling directory never reached the model even
though every install check passed.

So this reads the live transcripts instead. Namespaces are now
phase/method/dataset, so no question-text matching is needed: each recorded prompt
already says which cell it belongs to, and the only question is whether its
wording belongs to that cell's task.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Wording that asserts a task type. Regexes, so a phrase is counted once per
# prompt however it is spelled.
MARKERS: dict[str, list[str]] = {
    "maths": [r"\bmathematical problem\b", r"\bmath problem\b",
              r"\bsolve the (?:given )?math", r"\bcomplex math problem\b",
              r"\bGSM8K\b", r"highly skilled mathematician"],
    "boxed answer": [r"\\boxed", r"enclosed in .{0,8}oxed"],
    "code": [r"\bself-contained code\b", r"\bHumanEval\b",
             r"function signature and its docstring", r"\bPython code block\b"],
    "fixed 4 options": [r"4 answers enumerated as A, B, C and D",
                        r"one of the 4 letters", r"offered 4 is correct"],
    "multiple choice": [r"\bmultiple[- ]choice\b", r"\boption letter\b",
                        r"choose the correct answer"],
    "reading": [r"\bshortest exact span\b", r"\bthe passage\b"],
}

# What each dataset's own task is. Anything else found in its prompts is foreign,
# except entries listed in TOLERATED below.
NATIVE = {
    "math": {"maths", "boxed answer", "code"},      # Programmer is a maths tool here
    "amc": {"maths", "boxed answer", "code"},
    "mbpp": {"code"},
    "drop": {"reading"},
    "mmlu_pro": {"multiple choice", "boxed answer"},  # authors' MMLU node uses \boxed{A}
}

# Foreign wording that is deliberately left in place, with the reason. Printed as
# "declared" rather than counted as a problem, so the distinction between "known
# and argued" and "missed" stays visible.
TOLERATED = {
    ("drop", "code"): "the Programmer/CodeSolver operators genuinely write code to "
                      "count figures the passage states; that is the method's "
                      "operator, not a claim about the task",
    ("mmlu_pro", "code"): "same: the code operators are part of the workflow",
    ("mmlu_pro", "maths"): "G-Designer/CARD preserve the authors' MMLU ensemble's "
                           "Mathematician specialist role; this is role diversity, "
                           "not a statement that every item is a maths problem",
    # These three were flagged on the first run and each turned out to be the
    # detector matching something that is not prompt wording. Recorded rather than
    # deleted, so a future reader can see why they are not counted.
    ("mmlu_pro", "reading"): "matches the QUESTION text, not the prompt: MMLU-Pro "
                             "contains passage-based items ('The passage suggests "
                             "that the speaker would describe...'). Seen at 0.5-2.6% "
                             "on aflow, flowbank and masrouter, which is the rate of "
                             "such items in the split",
    ("math", "multiple choice"): "the authors' output_format carries a conditional "
                                 "clause, 'If it is a multiple choice question, "
                                 "please output the options' -- generic formatting "
                                 "they apply to their own datasets too, not a claim "
                                 "that this task is multiple choice",
    ("amc", "multiple choice"): "same conditional clause as math",
    ("drop", "multiple choice"): "the same authors' output_format clause, and the "
                                 "evidence that it is generic boilerplate rather "
                                 "than a claim about the task is that it appears at "
                                 "72.4% on masrouter/MATH -- one of the authors' own "
                                 "datasets. It is conditional ('If it is a multiple "
                                 "choice question'), so it asserts nothing false "
                                 "about DROP, and an output format is the method's "
                                 "design: adapting it per dataset would cross the "
                                 "line this whole audit exists to police. Observed "
                                 "effect: 99.4% of DROP replies still end 'Answer: "
                                 "<span>' with a real span",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", default="logs/transcripts.jsonl")
    parser.add_argument("--show", default=None,
                        help="method/dataset/kind to print an example of")
    parser.add_argument("--json", action="store_true",
                        help="compact machine-readable output, for the watchdog")
    # The transcript reaches gigabytes over a sweep. Scanning only the tail keeps
    # this cheap enough to run every watchdog cycle, and the tail is what answers
    # "is a cell that started recently contaminated" -- which is the question,
    # since a cell that has been clean for hours will not turn dirty.
    parser.add_argument("--tail-mb", type=int, default=0,
                        help="only scan the last N MB (0 = whole file)")
    args = parser.parse_args()

    compiled = {kind: [re.compile(p, re.IGNORECASE) for p in patterns]
                for kind, patterns in MARKERS.items()}
    totals: collections.Counter = collections.Counter()
    hits: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    examples: dict[tuple[str, str, str], str] = {}

    path = ROOT / args.transcripts
    with path.open("rb") as raw:
        if args.tail_mb:
            raw.seek(0, 2)
            window = min(raw.tell(), args.tail_mb * 1024 ** 2)
            raw.seek(-window, 2)
            raw.readline()  # discard a partial first line
        handle = (line.decode("utf-8", "replace") for line in raw)
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = (entry.get("namespace") or "").split("/")
            if len(parts) < 3:
                continue
            method, dataset = parts[1], parts[2]
            if dataset not in NATIVE:
                continue
            prompt = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
            flat = re.sub(r"\s+", " ", prompt)
            key = (method, dataset)
            totals[key] += 1
            for kind, patterns in compiled.items():
                if kind in NATIVE[dataset]:
                    continue
                for pattern in patterns:
                    match = pattern.search(flat)
                    if match:
                        hits[key][kind] += 1
                        spot = match.start()
                        examples.setdefault((method, dataset, kind),
                                            flat[max(0, spot - 90): spot + 110])
                        break

    findings = []
    for key in sorted(totals):
        method, dataset = key
        for kind, count in hits[key].items():
            if (dataset, kind) in TOLERATED:
                continue
            findings.append({"method": method, "dataset": dataset, "wording": kind,
                             "prompts": count, "of": totals[key],
                             "share": round(count / totals[key], 4)})
    if args.json:
        print(json.dumps({"findings": findings,
                          "cells": {f"{m}/{d}": n for (m, d), n in totals.items()}}))
        return

    print(f"  {'method':<26}{'dataset':<11}{'prompts':>9}   foreign wording (share)")
    problems = 0
    for key in sorted(totals):
        method, dataset = key
        found = hits[key]
        undeclared = {k: v for k, v in found.items() if (dataset, k) not in TOLERATED}
        declared = {k: v for k, v in found.items() if (dataset, k) in TOLERATED}
        detail = ", ".join(f"{k} {v / totals[key]:.1%}" for k, v in undeclared.items()) or "-"
        flag = "  <-- FIX" if undeclared else ""
        problems += len(undeclared)
        print(f"  {method:<26}{dataset:<11}{totals[key]:>9,}   {detail}{flag}")
        if declared:
            print(f"  {'':<37}   declared: "
                  + ", ".join(f"{k} {v / totals[key]:.1%}" for k, v in declared.items()))

    print(f"\n  {problems} undeclared (method, dataset, wording) combination(s)")
    if args.show:
        method, dataset, kind = args.show.split("/", 2)
        print(f"\n  === {args.show} ===\n  {examples.get((method, dataset, kind), 'none')}")
    else:
        for (method, dataset, kind), sample in sorted(examples.items()):
            if (dataset, kind) in TOLERATED:
                continue
            print(f"\n  {method}/{dataset} -- {kind}:\n    ...{sample}...")


if __name__ == "__main__":
    main()
