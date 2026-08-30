#!/usr/bin/env python3
"""Can G-Designer/CARD even say E through J on MMLU-Pro? And what does DROP get?

G-Designer ships three prompt domains, and the closest one to a ten-way
multiple-choice question is `mmlu` -- which hardcodes four options in four
separate places, including the final decision constraint. So this asks the
recorded transcripts a measurable question: what share of the gold answers lies
outside A-D, and what share of the method's own final answers ever lands there.

The DROP half is the same substitution seen from the other side: a reading task
inherits a prompt set that demands a single letter, so the completions show what
the decision node actually produced.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "shared")
import bench  # noqa: E402

ROOT = Path(__file__).resolve().parent
FOUR_OPTION = "4 answers enumerated as A, B, C and D"

gold: collections.Counter = collections.Counter()
for row in bench.load("mmlu_pro"):
    gold[chr(65 + row["answer_index"])] += 1

pred: collections.Counter = collections.Counter()
drop_samples: list[str] = []
drop_total = 0

for line in (ROOT / "logs" / "transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    namespace = entry.get("namespace") or ""
    if "gdesigner" not in namespace and "card" not in namespace:
        continue
    text = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
    if FOUR_OPTION not in text or "decision-maker" not in text:
        continue
    completion = str(entry.get("completion") or "")
    if "Answer: (X)" in text or "single option letter" in text:
        match = re.search(r"\(([A-J])\)|\b([A-J])\b", completion)
        pred[(match.group(1) or match.group(2)) if match else "none"] += 1
    elif "shortest exact span" in text:
        drop_total += 1
        if len(drop_samples) < 6:
            drop_samples.append(completion)

total_gold = sum(gold.values()) or 1
total_pred = sum(pred.values()) or 1

print(f"  {'letter':<8}{'gold share':>12}{'final answers given':>22}")
for letter in [chr(65 + i) for i in range(10)] + ["none"]:
    gold_share = gold.get(letter, 0) / total_gold * 100 if letter != "none" else 0.0
    pred_share = pred.get(letter, 0) / total_pred * 100
    flag = "   <-- gold answer the prompt forbids" if gold_share > 1 and pred_share == 0 else ""
    print(f"  {letter:<8}{gold_share:>11.1f}%{pred_share:>21.1f}%{flag}")

outside_gold = sum(v for k, v in gold.items() if k > "D") / total_gold
outside_pred = sum(v for k, v in pred.items() if k in set("EFGHIJ")) / total_pred
print(f"\n  gold answers outside A-D:  {outside_gold:.1%} of items")
print(f"  final answers outside A-D: {outside_pred:.1%} of {total_pred:,} decisions")

print(f"\n  === DROP final decisions under the same 4-letter constraint "
      f"({drop_total:,} calls) ===")
for sample in drop_samples:
    print(f"    {sample[:120]!r}")
