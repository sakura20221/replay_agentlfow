#!/usr/bin/env python3
"""Find prompts that name the wrong benchmark, inherited when a workspace was seeded.

Every SHARED_* workspace is a copy of one the authors shipped: SHARED_MBPP from
HumanEval, the rest from MATH. The code was adapted; the prose was not. One case is
already confirmed from a recorded exchange -- an MBPP item arriving with

    The previous solution failed some test cases in the HumanEval benchmark

so the model is told which benchmark it is on, and told wrong. That does not crash
anything, which is why it went unnoticed; it just quietly changes the task
description for one dataset and not the others.

Reports each mention with its file and the surrounding words, so a real mismatch is
distinguishable from a legitimate reference (an operator's own docstring, say).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Which benchmark each workspace legitimately is. A mention of any *other*
# benchmark name in its prompts is inherited wording.
EXPECTED = {
    "SHARED_MATH": {"math", "MATH"},
    "SHARED_AMC": {"math", "MATH", "AMC"},
    "SHARED_MBPP": {"MBPP", "mbpp"},
    "SHARED_DROP": {"DROP", "drop"},
    "SHARED_MMLUPRO": {"MMLU", "MMLU-Pro", "MMLU_Pro"},
}
FOREIGN = ["HumanEval", "GSM8K", "MBPP", "MATH", "DROP", "MMLU", "HotpotQA",
           "LiveCodeBench", "Codeforces"]

SEARCH_ROOTS = [
    "third_party/maas/maas/ext/maas/scripts/optimized",
    "third_party/daao/daao/ext/maas/scripts/optimized",
    "third_party/aflow/workspace",
    "third_party/flowbank/DiverseFlow/workspace",
]

findings = 0
for root in SEARCH_ROOTS:
    base = ROOT / root
    if not base.exists():
        continue
    for path in sorted(base.rglob("*.py")):
        parts = path.parts
        key = next((p for p in parts if p.startswith("SHARED_")), None)
        if key is None or key not in EXPECTED:
            continue
        # Only prose-bearing files: prompt text, not operator logic.
        if path.name not in ("prompt.py", "op_prompt.py", "prompt_custom.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        allowed = EXPECTED[key]
        for name in FOREIGN:
            if name in allowed:
                continue
            for match in re.finditer(re.escape(name), text):
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                context = " ".join(text[start:end].split())
                findings += 1
                print(f"\n  {path.relative_to(ROOT)}")
                print(f"    mentions {name!r} but the workspace is {key}")
                print(f"    ...{context}...")
                break          # one example per file per foreign name is enough

print(f"\n  {findings} contaminated mention(s)")
