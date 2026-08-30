#!/usr/bin/env python3
"""How many of our MBPP items land on an entry the authors deliberately blanked.

`extract_test_cases_from_jsonl` carries a `hardcoded_cases` dict whose values are
empty strings. An empty string is not the failure mode that discarded 1,020 samples
-- `for case in ""` iterates zero times and raises nothing -- but it does mean the
workflow's own self-check runs no tests for that item. Worth counting rather than
waving away, and worth distinguishing from None in any test that checks this path.
"""
import json
import re
from pathlib import Path

ours = set()
for name in ("mbpp.jsonl", "mbpp_search.jsonl"):
    for line in Path("shared/data", name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ours.add(json.loads(line)["entry_point"])

for repo in ("maas", "daao"):
    source = Path(f"third_party/{repo}/{repo}/ext/maas/scripts/utils.py")
    text = source.read_text(encoding="utf-8")
    start = text.index("CodeDataset.MBPP.value")
    block = text[start: text.index("}", start)]
    blanked = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)":\s*""', block)
    overlap = sorted(set(blanked) & ours)
    print(f"  {repo}: {len(blanked)} entry point(s) blanked by the authors; "
          f"{len(overlap)} of our {len(ours)} items land on one "
          f"({100 * len(overlap) / len(ours):.2f}%)")
    if overlap:
        print(f"      {overlap}")
