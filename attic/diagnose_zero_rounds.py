"""Why did a round score zero? Read the round's own per-item records.

Three outcomes are indistinguishable from the score alone and need different
fixes, so they are separated here:

  * the graph raised          -> prediction is empty or an error string
  * the answer was unscoreable -> prediction has text but no extractable letter
  * the model was simply wrong  -> prediction has a letter, it disagrees with gold
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

WORKSPACE = sys.argv[1] if len(sys.argv) > 1 else \
    "third_party/aflow/workspace/SHARED_MMLUPRO/workflows"
sys.path.insert(0, "shared")
import bench  # noqa: E402

results = json.loads(Path(WORKSPACE, "results.json").read_text(encoding="utf-8"))
by_round = {}
for entry in results:
    by_round.setdefault(int(entry["round"]), []).append(entry.get("score"))

for round_number in sorted(by_round):
    scores = [s for s in by_round[round_number] if isinstance(s, (int, float))]
    csvs = sorted(glob.glob(f"{WORKSPACE}/round_{round_number}/*.csv"))
    graph = Path(WORKSPACE, f"round_{round_number}", "graph.py")
    call = ""
    if graph.exists():
        found = re.findall(r"(?:await self\.\w+\([^)]*\))", graph.read_text(encoding="utf-8"))
        call = " | ".join(found[:3])[:110]
    print(f"\n=== round {round_number}  score={scores}  ({len(csvs)} csv) ===")
    print(f"    calls: {call}")
    if not csvs:
        print("    no per-item records")
        continue
    rows = list(csv.DictReader(open(csvs[-1], encoding="utf-8")))
    empty = unscoreable = wrong = right = 0
    samples = []
    for row in rows:
        prediction = (row.get("prediction") or "").strip()
        if not prediction or prediction.lower() in {"none", "nan"}:
            empty += 1
            if len(samples) < 2:
                samples.append(("EMPTY", prediction[:120]))
            continue
        letter = bench._extract_mmlu_pro_letter(prediction, 10)
        if not letter:
            unscoreable += 1
            if len(samples) < 4:
                samples.append(("NO LETTER", prediction[-160:]))
        elif float(row.get("score") or 0) > 0:
            right += 1
        else:
            wrong += 1
            if len(samples) < 4:
                samples.append((f"WRONG letter={letter} gold={row.get('expected_output')}",
                                prediction[-100:]))
    print(f"    {len(rows)} items: empty={empty} unscoreable={unscoreable} "
          f"wrong={wrong} right={right}")
    for kind, text in samples:
        print(f"      [{kind}] {text!r}")
