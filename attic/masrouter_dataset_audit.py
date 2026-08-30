#!/usr/bin/env python3
"""Per dataset, what is MasRouter actually being told, and what does it answer?

MasRouter composes each agent's prompt from four independent layers, and any one of
them can carry another dataset's task identity:

  1. the task type it routes to        (Math / Commonsense / Code)
  2. that type's role pool             MAR/Roles/<Type>/*.json
  3. the role's output format          MAR/Prompts/output_format.py
  4. the final decision node's prompt  MAR/Roles/FinalNode/*.json

Layers 1, 2 and 4 were adapted; this reads the live transcripts to check what
reached the model on each dataset, and what shape the answers came back in --
because an instruction only matters if the reply shows it.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Phrases that identify each layer in a recorded prompt.
LAYER_MARKERS = {
    "output_format:Answer": "The last line of your output must contain only the final result",
    "output_format:multiple-choice clause": "If it is a multiple choice question, please output the options",
    "output_format:CodeCompletion": "You will be given a function signature and its docstring",
    "output_format:CodeSolver": "Analyze the question and write functions to solve the problem",
    "output_format:Keys": "Please provide relevant keywords",
    "role:span wording (adapted)": "quote the shortest exact span from the passage",
    "role:choose the correct answer": "choose the correct answer",
    "role:complex math problem": "complex math problem",
    "final:boxed maths": "You will be given a math problem, analysis and code",
    "final:option letters (adapted)": "one of the option letters offered with the question",
    "final:span (adapted)": "shortest exact span from the passage, copied rather than paraphrased",
    "final:code block": "Use a Python code block to write your response",
}

ANSWER_SHAPES = (
    ("The answer is <number>", r"[Tt]he answer is\s+-?[\d.,/]+\s*$"),
    ("The answer is <letter>", r"[Tt]he answer is\s+\(?[A-J]\)?\s*[.]?\s*$"),
    ("boxed", r"\\boxed\{"),
    ("Answer: line", r"(?m)^\s*Answer:\s*\S"),
    ("code block", r"```python"),
    ("The answer is <text>", r"[Tt]he answer is\s+\S+"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", default="logs/transcripts.jsonl")
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()

    layers: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    shapes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    totals: collections.Counter = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)

    path = ROOT / args.transcripts
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open(errors="replace") as handle:
        for line in handle:
            if "masrouter" not in line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            namespace = entry.get("namespace") or ""
            # Namespaces are phase/method/dataset, so the dataset is the last part.
            parts = namespace.split("/")
            if len(parts) < 3 or parts[1] != "masrouter":
                continue
            dataset = parts[2]
            prompt = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
            completion = str(entry.get("completion") or "")
            totals[dataset] += 1
            for label, needle in LAYER_MARKERS.items():
                if needle in prompt:
                    layers[dataset][label] += 1
            for label, pattern in ANSWER_SHAPES:
                if re.search(pattern, completion.strip()):
                    shapes[dataset][label] += 1
                    break
            else:
                shapes[dataset]["other"] += 1
            if len(samples[dataset]) < args.examples and completion.strip():
                samples[dataset].append(completion.strip()[-120:])

    if not totals:
        raise SystemExit("no masrouter traffic recorded yet")

    for dataset in sorted(totals):
        print(f"\n  === {dataset}  ({totals[dataset]:,} calls) ===")
        print("    layers present in the prompt:")
        for label, count in layers[dataset].most_common():
            share = count / totals[dataset]
            print(f"      {label:<38}{count:>7,}  {share:>6.1%}")
        print("    reply shapes:")
        for label, count in shapes[dataset].most_common():
            print(f"      {label:<38}{count:>7,}  {count / totals[dataset]:>6.1%}")
        for sample in samples[dataset]:
            print(f"      tail: {sample!r}")


if __name__ == "__main__":
    main()
