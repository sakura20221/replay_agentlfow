#!/usr/bin/env python3
"""Every prompt every method sends, checked for cross-domain wording.

The first scan was too narrow: it looked only at prompt.py / op_prompt.py under
SHARED_* directories, which misses the G-Designer family and MasRouter (whose
prompts live in the shim's own modules), the operator files that embed prose, and
any generated round's prompt.

Two kinds of contamination are reported separately, because they need different
judgements:

  NAMES THE WRONG BENCHMARK  -- the model is told which dataset it is on, wrongly.
                                Unambiguous; must be fixed.
  WRONG TASK VOCABULARY      -- e.g. "solve the mathematical problem" reaching a
                                multiple-choice dataset. Needs a look before
                                editing: some methods legitimately reason about a
                                question as a maths problem.

Recorded prompts are the ground truth for what was actually sent, so the last
section cross-checks the files against logs/transcripts.jsonl.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BENCHMARKS = ["HumanEval", "GSM8K", "MBPP", "MATH", "DROP", "MMLU-Pro", "MMLU_Pro",
              "MMLU", "HotpotQA", "LiveCodeBench", "Codeforces", "AMC", "GPQA"]

# What each dataset legitimately is, and the vocabulary that fits it.
LEGITIMATE = {
    "math": {"MATH", "AMC", "GSM8K"},
    "amc": {"AMC", "MATH", "GSM8K"},
    "mbpp": {"MBPP"},
    "drop": {"DROP"},
    "mmlu_pro": {"MMLU", "MMLU-Pro", "MMLU_Pro"},
}
KEY_TO_DATASET = {"SHARED_MATH": "math", "SHARED_AMC": "amc", "SHARED_MBPP": "mbpp",
                  "SHARED_DROP": "drop", "SHARED_MMLUPRO": "mmlu_pro"}

# Task vocabulary that does not belong to a dataset.
WRONG_VOCAB = {
    "mmlu_pro": [r"\bmathematical problem\b", r"\bsolve the (?:given )?math",
                 r"\bwrite (?:a |the )?(?:python )?function\b", r"\bcode block\b"],
    "drop": [r"\bmathematical problem\b", r"\bwrite (?:a |the )?(?:python )?function\b"],
    "mbpp": [r"\bmathematical problem\b", r"\bmultiple[- ]choice\b"],
    "math": [r"\bmultiple[- ]choice\b", r"\bwrite (?:a |the )?(?:python )?function\b"],
    "amc": [r"\bmultiple[- ]choice\b"],
}

PROSE_FILES = {"prompt.py", "op_prompt.py", "prompt_custom.py", "operator.py",
               "shared_prompt_sets.py", "prompt_set.py", "prompts.py"}

SEARCH_ROOTS = [
    "third_party/maas/maas/ext/maas/scripts/optimized",
    "third_party/daao/daao/ext/maas/scripts/optimized",
    "third_party/aflow/workspace",
    "third_party/flowbank/DiverseFlow/workspace",
    "third_party/gdesigner",
    "third_party/card",
    "third_party/masrouter",
]

wrong_name: list[tuple[str, str, str, str]] = []
wrong_vocab: list[tuple[str, str, str, str]] = []

for root in SEARCH_ROOTS:
    base = ROOT / root
    if not base.exists():
        continue
    for path in sorted(base.rglob("*.py")):
        if path.name not in PROSE_FILES:
            continue
        if "__pycache__" in path.parts or "/.git/" in str(path):
            continue
        key = next((p for p in path.parts if p in KEY_TO_DATASET), None)
        dataset = KEY_TO_DATASET.get(key) if key else None
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = str(path.relative_to(ROOT))

        if dataset:
            allowed = LEGITIMATE[dataset]
            for name in BENCHMARKS:
                if name in allowed or name not in text:
                    continue
                match = re.search(re.escape(name), text)
                context = " ".join(text[max(0, match.start() - 55):match.end() + 55].split())
                wrong_name.append((relative, dataset, name, context))
                break
            for pattern in WRONG_VOCAB.get(dataset, []):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    context = " ".join(text[max(0, match.start() - 55):match.end() + 55].split())
                    wrong_vocab.append((relative, dataset, pattern, context))

print("  ### prompts that name the wrong benchmark ###")
for relative, dataset, name, context in wrong_name:
    print(f"\n  {relative}\n    dataset={dataset} names {name!r}\n    ...{context}...")
print(f"\n  {len(wrong_name)} file(s)\n")

print("  ### prompts whose task vocabulary does not fit the dataset ###")
seen = set()
for relative, dataset, pattern, context in wrong_vocab:
    key = (relative, pattern)
    if key in seen:
        continue
    seen.add(key)
    print(f"\n  {relative}\n    dataset={dataset} matches {pattern}\n    ...{context}...")
print(f"\n  {len(seen)} file(s)")

# What was actually sent, per dataset, from the recorded exchanges.
print("\n  ### cross-check against recorded prompts ###")
counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
transcripts = ROOT / "logs" / "transcripts.jsonl"
if transcripts.exists():
    for line in transcripts.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
        method = (entry.get("namespace") or "?").split("/")[-1]
        for name in ("HumanEval", "GSM8K"):
            if name in text:
                counts[method][name] += 1
    for method, tally in sorted(counts.items()):
        print(f"    {method:<12} {dict(tally)}")
    if not counts:
        print("    no recorded prompt mentions HumanEval or GSM8K")
