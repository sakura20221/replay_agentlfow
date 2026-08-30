"""Which repos can look up MBPP public tests for our items, and which cannot.

The MaAS-family `Test` operator calls extract_test_cases_from_jsonl(entry_point)
and iterates the result without checking it. A name missing from the lookup file
returns None, so the sample raises instead of being scored -- 147 discarded in
daao/mbpp before this was noticed.
"""
import json
from pathlib import Path

FILES = {
    "aflow": "third_party/aflow/data/datasets/mbpp_public_test.jsonl",
    "flowbank": "third_party/flowbank/DiverseFlow/data/datasets/mbpp_public_test.jsonl",
    "maas": "third_party/maas/maas/ext/maas/data/mbpp_public_test.jsonl",
    "daao": "third_party/daao/daao/ext/maas/data/mbpp_public_test.jsonl",
}

splits = {}
for name in ("mbpp", "mbpp_search"):
    rows = [json.loads(l) for l in Path(f"shared/data/{name}.jsonl").read_text().splitlines() if l.strip()]
    splits[name] = [r.get("entry_point") for r in rows]

for repo, path in FILES.items():
    p = Path(path)
    if not p.exists():
        print(f"  {repo:<10} MISSING {path}")
        continue
    names = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(json.loads(line).get("entry_point"))
    parts = []
    for split, eps in splits.items():
        missing = sum(1 for e in eps if e not in names)
        parts.append(f"{split}: {len(eps) - missing}/{len(eps)} found"
                     f" ({100 * missing / len(eps):.0f}% missing)")
    print(f"  {repo:<10} {len(names):>4} names   " + "   ".join(parts))
