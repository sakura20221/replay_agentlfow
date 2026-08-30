#!/usr/bin/env python3
"""Which finished jobs have scores that the looping problem could actually move.

The global rate (0.46% of calls) says nothing about a specific cell in the table.
What matters per job is: how many of *its* evaluation items had at least one call
that hit the cap, because those are the items whose answer came from the recovery
path rather than from the workflow.

Attribution needs both axes at once -- the method comes from the proxy namespace,
the item from matching the question text inside the prompt -- so it is done here
rather than inferred from either alone.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "shared")
import bench  # noqa: E402

CAP = 8192

# Distinctive middle slice of each evaluation item's question. The head is often a
# shared preamble and the tail is the answer-format instruction every item carries,
# so neither identifies an item.
signatures: dict[str, tuple[str, str]] = {}
split_size: dict[str, int] = {}
for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
    rows = bench.load(dataset)
    split_size[dataset] = len(rows)
    for row in rows:
        text = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()
        if len(text) < 80:
            continue
        middle = text[len(text) // 3: len(text) // 3 + 60]
        if len(middle) == 60:
            signatures.setdefault(middle, (dataset, str(row["uid"])))

print(f"  indexed {len(signatures):,} question slices", flush=True)

touched: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
capped_calls: dict[str, int] = collections.Counter()
unmatched = 0

for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (entry.get("completion_tokens") or 0) < CAP - 200:
        continue
    method = (entry.get("namespace") or "?").split("/")[-1]
    capped_calls[method] += 1
    prompt = re.sub(r"\s+", " ",
                    " ".join(str(m.get("content", ""))
                             for m in (entry.get("messages") or [])))
    for slice_, (dataset, uid) in signatures.items():
        if slice_ in prompt:
            touched[(method, dataset)].add(uid)
            break
    else:
        unmatched += 1

print(f"  capped calls by method: {dict(capped_calls)}")
print(f"  capped calls that matched no evaluation item: {unmatched:,} "
      f"(search-split items and optimiser calls)\n")

print(f"  {'method':<12}{'dataset':<10}{'items touched':>15}{'split':>8}{'share':>9}")
rows = sorted(touched.items(), key=lambda kv: -len(kv[1]) / split_size[kv[0][1]])
for (method, dataset), uids in rows:
    total = split_size[dataset]
    share = len(uids) / total
    flag = "  <-- worth re-examining" if share >= 0.03 else ""
    print(f"  {method:<12}{dataset:<10}{len(uids):>15,}{total:>8,}{share:>8.2%}{flag}")
if not rows:
    print("  no capped call matched an evaluation item")
