#!/usr/bin/env python3
"""For every (method, dataset) cell: whose prompts is it actually using?

Two independent views, because either alone misleads:

STRUCTURE -- which source workspace the prompt files were copied from. Cheap and
complete, but a copy is only a problem if its wording does not fit the new dataset.

EVIDENCE -- what was really sent, taken from logs/transcripts.jsonl and attributed
to a dataset by matching the question text. This is the view that matters, and it
catches wording that names no benchmark at all (a DROP item asked to "solve the
mathematical problem" mentions no dataset name, so a name-based scan misses it).
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

# Vocabulary that belongs to a task type. A prompt reaching the wrong dataset shows
# up as its type not matching.
MARKERS = {
    "maths": [r"\bmathematical problem\b", r"\bsolve the (?:given )?math",
              r"GSM8K", r"\bMATH benchmark\b"],
    "code": [r"\bself-contained code\b", r"\bpython (?:function|code)\b",
             r"HumanEval", r"\bcode block\b"],
    "multiple choice": [r"\bmultiple[- ]choice\b", r"\boption letter\b"],
    "reading": [r"\bpassage\b", r"\bshortest exact span\b"],
}
# The type each dataset actually is.
NATIVE = {"math": "maths", "amc": "maths", "mbpp": "code",
          "drop": "reading", "mmlu_pro": "multiple choice"}

print("  ### STRUCTURE: which workspace each cell's prompts were copied from ###")
# The MaAS-family installer records this mapping; AFlow/DiverseFlow seed per key.
for label, install in (("maas/daao", "shims/maas_family/install.py"),
                       ("aflow", "shims/aflow/install.py"),
                       ("flowbank", "shims/diverseflow/install.py")):
    path = ROOT / install
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    match = re.search(r"WORKSPACE_SOURCE\s*=\s*\{(.*?)\}", text, re.DOTALL)
    mapping = dict(re.findall(r'"(SHARED_[A-Z]+)":\s*"([A-Za-z_]+)"',
                              match.group(1) if match else ""))
    print(f"    {label:<12} {mapping or 'no WORKSPACE_SOURCE table'}")

print("\n  ### EVIDENCE: task vocabulary in the prompts actually sent ###")

# Question-text signature -> dataset, for attributing a recorded prompt.
signatures: dict[str, str] = {}
for dataset in NATIVE:
    for row in bench.load(dataset):
        text = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()
        if len(text) < 80:
            continue
        middle = text[len(text) // 3: len(text) // 3 + 60]
        if len(middle) == 60:
            signatures.setdefault(middle, dataset)

cells: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
totals: dict[tuple[str, str], int] = collections.Counter()

for line in (ROOT / "logs" / "transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    prompt = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
    prompt = re.sub(r"\s+", " ", prompt)
    dataset = None
    for slice_, name in signatures.items():
        if slice_ in prompt:
            dataset = name
            break
    if dataset is None:
        continue
    method = (entry.get("namespace") or "?").split("/")[-1]
    key = (method, dataset)
    totals[key] += 1
    native = NATIVE[dataset]
    for kind, patterns in MARKERS.items():
        if kind == native:
            continue
        for pattern in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                cells[key][kind] += 1
                break

print(f"    {'method':<12}{'dataset':<11}{'prompts':>9}{'foreign vocabulary':>34}")
for key in sorted(totals, key=lambda k: (-sum(cells[k].values()) / max(totals[k], 1), k)):
    method, dataset = key
    foreign = cells[key]
    share = sum(foreign.values()) / max(totals[key], 1)
    detail = ", ".join(f"{k} {v}" for k, v in foreign.most_common()) or "-"
    flag = "  <-- contaminated" if share > 0.05 else ""
    print(f"    {method:<12}{dataset:<11}{totals[key]:>9,}   {share:>6.1%}  {detail}{flag}")
