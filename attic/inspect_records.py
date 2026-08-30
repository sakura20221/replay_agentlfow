"""Why do the G-Designer-family records not match the evaluation split?"""
import json
import re
import sys

sys.path.insert(0, "shared")
import bench

path = sys.argv[1]
dataset = sys.argv[2]
records = json.load(open(path, encoding="utf-8"))
print(f"  {len(records)} records; fields: {list(records[0].keys())}")

record = records[-1]
question = record.get("Question")
text = question if isinstance(question, str) else json.dumps(question, ensure_ascii=False)
print(f"  Question type: {type(question).__name__}, {len(text)} chars")
print(f"  record  head: {text[:130]!r}")
print(f"  record  tail: {text[-90:]!r}")
print(f"  has the ICL example: {'Format example' in text}")
print(f"  Solved: {record.get('Solved')}")

row = bench.load(dataset)[0]
mine = bench.question_text(dataset, row)
print(f"  bench   head: {mine[:130]!r}")
print(f"  bench   tail: {mine[-90:]!r}")

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()[:200]
keys = {norm(bench.question_text(dataset, r)) for r in bench.load(dataset)}
print(f"\n  eval keys: {len(keys)}   record key in keys: {norm(text) in keys}")
# Where do they diverge?
matched = sum(1 for r in records
              if norm(r.get('Question') if isinstance(r.get('Question'), str)
                      else json.dumps(r.get('Question'), ensure_ascii=False)) in keys)
print(f"  matched records: {matched}/{len(records)}")
a, b = norm(text), sorted(keys)[0]
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        print(f"  first divergence from an arbitrary eval key at char {i}: {a[i:i+40]!r} vs {b[i:i+40]!r}")
        break
