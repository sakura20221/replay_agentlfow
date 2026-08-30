"""Bad-case harvest for the top-3 methods.

Per dataset, aligns daao / maas per-item results by gold answer and reports:
  - all3_wrong : nobody solved it  -> model ceiling / task-inherent hardness
  - split      : some solved, some not -> orchestration matters, optimisation room
Dumps concrete cases for eyeballing.
"""
import csv, json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SWEEP_RUNS", str(ROOT / "runs_v5"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))
os.chdir(ROOT)

import bench
import importlib.util
spec = importlib.util.spec_from_file_location("collectmod", ROOT / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

csv.field_size_limit(sys.maxsize)
SHARED_KEY = {"math": "SHARED_MATH", "amc": "SHARED_AMC", "mbpp": "SHARED_MBPP",
              "drop": "SHARED_DROP", "mmlu_pro": "SHARED_MMLUPRO"}
DATASETS = ["math", "amc", "mbpp", "drop", "mmlu_pro"]
METHODS = ["daao", "maas"]


def load_cell(method, dataset):
    key = SHARED_KEY[dataset]
    base = ROOT / "third_party" / method / method / "ext" / "maas" / "scripts" / "optimized" / key / "test"
    job = ROOT / "runs_v5" / method / dataset / "repeat1"
    floor = 0.0
    for marker in ("test.cmd", "search.cmd"):
        p = job / marker
        if p.exists():
            floor = p.stat().st_mtime
            break
    files = [f for f in base.glob("round_*/0.*.csv") if f.stat().st_mtime >= floor]
    if not files:
        return {}
    path = max(files, key=lambda p: p.stat().st_mtime)
    out = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as h:
        for i, row in enumerate(csv.DictReader(h)):
            gold = (row.get("expected_output") or "").strip()
            if not gold:
                continue
            grading = collect._grading_row(dataset, gold)
            if grading is None:
                continue
            try:
                value, extracted = bench.score(dataset, grading, row.get("prediction") or "")
            except Exception:
                continue
            q = (row.get("question") or row.get("inputs") or "")
            out[gold + "||" + str(q)[:120]] = {
                "score": value, "gold": gold, "q": str(q)[:400],
                "pred": str(row.get("prediction") or "")[:500],
                "extracted": str(extracted)[:120],
            }
    return out


def main():
    outdir = ROOT / "audits" / "badcases"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in DATASETS:
        cells = {m: load_cell(m, ds) for m in METHODS}
        keys = set()
        for m in METHODS:
            keys |= set(cells[m])
        both_wrong, split = [], []
        for k in keys:
            scores = {m: cells[m].get(k, {}).get("score") for m in METHODS}
            vals = [v for v in scores.values() if v is not None]
            if len(vals) < 2:
                continue
            if all(v <= 0.001 for v in vals):
                both_wrong.append((k, scores, cells["daao"].get(k) or cells["maas"].get(k)))
            elif any(v <= 0.001 for v in vals) and any(v > 0.5 for v in vals):
                split.append((k, scores, cells["daao"].get(k), cells["maas"].get(k)))
        report[ds] = {"aligned": len(keys), "both_wrong": len(both_wrong),
                      "split": len(split)}
        print(f"{ds:10} aligned={len(keys):5}  both_wrong={len(both_wrong):4}  "
              f"split(one solved,one not)={len(split):4}", flush=True)
        dump = {
            "both_wrong": [{"gold": c["gold"], "q": c["q"], "pred_tail": c["pred"][-300:],
                            "extracted": c["extracted"]}
                           for _, _, c in both_wrong[:60] if c],
            "split": [{"gold": (d or m)["gold"], "q": (d or m)["q"],
                       "daao_score": s.get("daao"), "maas_score": s.get("maas"),
                       "daao_extracted": (d or {}).get("extracted"),
                       "maas_extracted": (m or {}).get("extracted")}
                      for _, s, d, m in split[:60]],
        }
        (outdir / f"{ds}.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("BADCASE_DONE")


if __name__ == "__main__":
    main()
