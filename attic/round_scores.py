"""Per-round scores for every searched workspace, so a silently-all-zero run shows.

Checking `status == ok` is not enough: a graph that raises inside every item is
caught by AFlow, scored 0, and exits cleanly -- eight rounds at exactly 0.0000 with
"convergence detected" is what that looks like from the outside.
"""
import json
from pathlib import Path

ROOT = Path(".")
for results in sorted(ROOT.glob("third_party/*/workspace/SHARED_*/workflows/results.json")) + \
              sorted(ROOT.glob("third_party/*/*/workspace/SHARED_*/workflows/results.json")):
    try:
        data = json.loads(results.read_text(encoding="utf-8"))
    except Exception:
        continue
    scores = [r.get("score") for r in data]
    numeric = [s for s in scores if isinstance(s, (int, float))]
    nonzero = [s for s in numeric if s > 0]
    label = "/".join(results.parts[1:-2]).replace("workspace/", "")
    flag = ""
    if len(numeric) > 2 and len(nonzero) <= 1:
        flag = "  <-- 只有种子轮非零,疑似静默全零"
    print(f"  {label:<52} {len(numeric):>2} 轮  非零 {len(nonzero):>2}  "
          f"最高 {max(numeric) if numeric else 0:.4f}{flag}")
