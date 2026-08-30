"""Is every method's DROP number on the same scale?

DROP is scored by token-level F1, so a correct-but-differently-worded span earns a
fraction. If one repo records the F1 and another binarises it, their averages are
not comparable -- and the gap would look like a method difference.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "shared")
sys.path.insert(0, ".")
from collect import RUN_TAG, record_uid  # noqa: E402

for repo, method in (("gdesigner", "gdesigner"), ("card", "card"),
                     ("gdesigner", "gdesigner_authordefault"),
                     ("card", "card_authordefault")):
    path = Path(f"third_party/{repo}/result/{RUN_TAG}/{method}_drop_r1.json")
    if not path.exists():
        continue
    records = json.loads(path.read_text(encoding="utf-8"))
    values = [float(r.get("Solved") or 0.0) for r in records]
    hist = collections.Counter(round(v, 2) for v in values)
    fractional = sum(1 for v in values if 0.0 < v < 1.0)
    print(f"  {method:<26} {len(values)} records  fractional={fractional} "
          f"({100 * fractional / len(values):.1f}%)")
    print(f"      value histogram (top 6): {hist.most_common(6)}")
