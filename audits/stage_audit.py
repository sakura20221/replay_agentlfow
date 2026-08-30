#!/usr/bin/env python3
"""For every (method, dataset), print what each STAGE actually instructs, and what
comes back -- so dataset fit can be read rather than inferred from a regex.

The stages are not enumerated by hand. Within one cell the instruction text repeats
across calls and the question text does not, so lines that appear in many of a
cell's prompts are its instructions, and the set of instruction lines a prompt
carries identifies which stage produced it. That means a stage nobody thought to
look for still shows up, which is the point: the two contamination bugs found so
far were both in stages I had not enumerated (MaAS's op_prompt.py constants, and
MasRouter's role profiles).

    python stage_audit.py --method masrouter
    python stage_audit.py --method maas --dataset drop --lines 14
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Lines shorter than this are punctuation, brackets or fragments of a question.
MIN_LINE = 25
# A line in at least this share of a cell's prompts is instruction, not content.
INSTRUCTION_SHARE = 0.04


def normalise(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def load(path: Path, method: str | None, dataset: str | None) -> dict:
    cells: dict[tuple[str, str], list[tuple[list[str], str]]] = collections.defaultdict(list)
    with path.open(errors="replace") as handle:
        for raw in handle:
            if method and method not in raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            parts = (entry.get("namespace") or "").split("/")
            if len(parts) < 3:
                continue
            phase, entry_method, entry_dataset = parts[0], parts[1], parts[2]
            if method and entry_method != method:
                continue
            if dataset and entry_dataset != dataset:
                continue
            prompt = "\n".join(str(m.get("content", ""))
                               for m in (entry.get("messages") or []))
            lines = [normalise(line) for line in prompt.splitlines()]
            lines = [line for line in lines if len(line) >= MIN_LINE]
            cells[(entry_method, entry_dataset)].append(
                (lines, str(entry.get("completion") or "")))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", default="logs/transcripts.jsonl")
    parser.add_argument("--method", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--stages", type=int, default=6, help="stages per cell")
    parser.add_argument("--lines", type=int, default=10, help="instruction lines per stage")
    parser.add_argument("--width", type=int, default=155)
    args = parser.parse_args()

    cells = load(ROOT / args.transcripts, args.method, args.dataset)
    if not cells:
        raise SystemExit("no matching traffic recorded yet")

    for (method, dataset), records in sorted(cells.items()):
        frequency: collections.Counter = collections.Counter()
        for lines, _completion in records:
            frequency.update(set(lines))
        threshold = max(2, int(len(records) * INSTRUCTION_SHARE))
        instructions = {line for line, count in frequency.items() if count >= threshold}

        # Stage identity: which instruction lines this prompt carries.
        stages: dict[frozenset, list[str]] = collections.defaultdict(list)
        for lines, completion in records:
            key = frozenset(line for line in lines if line in instructions)
            if key:
                stages[key].append(completion)

        print(f"\n{'=' * 100}")
        print(f"### {method} / {dataset}   {len(records):,} calls, "
              f"{len(stages)} distinct stage signature(s)")
        for key, completions in sorted(stages.items(), key=lambda kv: -len(kv[1]))[: args.stages]:
            # Order the lines as they appear in frequency, most common first: the
            # shared boilerplate comes first, the stage-specific text after.
            ordered = sorted(key, key=lambda line: -frequency[line])
            print(f"\n  --- stage: {len(completions):,} calls "
                  f"({len(completions) / len(records):.0%}) ---")
            for line in ordered[: args.lines]:
                print(f"      | {line[: args.width]}")
            if len(ordered) > args.lines:
                print(f"      | ... {len(ordered) - args.lines} more instruction line(s)")
            tails = [c.strip()[-90:].replace("\n", " ⏎ ") for c in completions if c.strip()]
            for tail in tails[:2]:
                print(f"      > reply tail: {tail}")
            if not tails:
                print("      > reply tail: (all empty)")


if __name__ == "__main__":
    main()
