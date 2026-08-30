#!/usr/bin/env python3
"""What task type does MasRouter actually assign to DROP and MMLU-Pro items?

MasRouter does not take a per-dataset prompt set: it carries a list of task
profiles (Math, Commonsense, Code) and decides at runtime which one a question is,
then picks a collaboration pattern and roles to match. So the question is not
"which dataset's prompts did it inherit" but "which of its three types did it
choose, for datasets that match none of them".

Answered from the recorded prompts, since the choice appears in the text it sends.
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

# Attribute a prompt to a dataset by a distinctive slice of the question.
signatures: dict[str, str] = {}
for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
    for row in bench.load(dataset):
        text = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()
        if len(text) < 80:
            continue
        middle = text[len(text) // 3: len(text) // 3 + 60]
        if len(middle) == 60:
            signatures.setdefault(middle, dataset)

TASK_NAMES = ("Math", "Commonsense", "Code")
by_dataset: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
examples: dict[str, dict] = {}

for line in (ROOT / "logs" / "transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (entry.get("namespace") or "") != "train/masrouter":
        continue
    prompt = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
    flat = re.sub(r"\s+", " ", prompt)
    dataset = None
    for slice_, name in signatures.items():
        if slice_ in flat:
            dataset = name
            break
    if dataset is None:
        continue
    # Which of its task profiles is named in the prompt, and which role.
    for name in TASK_NAMES:
        if re.search(rf"\b{name}\b", flat):
            by_dataset[dataset][name] += 1
    role = re.search(r"You are an? ([A-Z][A-Za-z ]{3,40})", prompt)
    if role:
        by_dataset[dataset]["role:" + role.group(1).strip()] += 1
    examples.setdefault(dataset, entry)

print(f"  {'dataset':<11}{'task profile mentions and roles'}")
for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
    tally = by_dataset.get(dataset)
    if not tally:
        print(f"  {dataset:<11}no attributed prompt")
        continue
    types = {k: v for k, v in tally.items() if not k.startswith("role:")}
    roles = {k[5:]: v for k, v in tally.items() if k.startswith("role:")}
    print(f"  {dataset:<11}types={types}")
    print(f"  {'':<11}roles={dict(collections.Counter(roles).most_common(4))}")

for dataset in ("drop", "mmlu_pro"):
    entry = examples.get(dataset)
    if not entry:
        continue
    print(f"\n  === one real {dataset} prompt MasRouter sent ===")
    for message in (entry.get("messages") or [])[:2]:
        content = str(message.get("content", ""))
        print(f"    [{message.get('role')}] {content[:400]!r}")
