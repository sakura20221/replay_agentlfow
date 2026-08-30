"""Audit every awarded answer for the top-3 methods (daao, maas, flowbank).

Splits awarded items into:
  exact  - extracted answer literally equals gold (no judgement involved)
  equiv  - full credit via equivalence layers (DUMPED for manual review)
  partial- 0<score<1 (DROP official F1 partial credit)
  exec   - mbpp: full credit = test suite passed on execution (no string notion)
"""
import csv, json, os, sys
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


def rows_maas(method, dataset):
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
        return []
    path = max(files, key=lambda p: p.stat().st_mtime)
    out = []
    with path.open(newline="", encoding="utf-8", errors="replace") as h:
        for row in csv.DictReader(h):
            gold = (row.get("expected_output") or "").strip()
            if gold:
                out.append((gold, row.get("prediction") or ""))
    return out


def rows_flowbank(dataset):
    key = SHARED_KEY[dataset]
    ws = ROOT / "third_party" / "flowbank" / "DiverseFlow" / "workspace" / key / "workflows"
    out = []
    for rd in sorted(ws.glob("round_*/test_results")):
        files = sorted(rd.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            continue
        with files[-1].open(encoding="utf-8", errors="replace") as h:
            for line in h:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                gold = str(r.get("expected_output") or "").strip()
                if gold:
                    out.append((gold, str(r.get("prediction") or "")))
    return out


def norm(s):
    return " ".join(str(s).split()).lower()


def audit(method, dataset, pairs):
    stats = {"cell": f"{method}/{dataset}", "items": len(pairs), "exact": 0,
             "equiv": 0, "partial": 0, "exec_pass": 0, "zero": 0, "grade_fail": 0}
    dumps = []
    for gold, pred in pairs:
        grading = collect._grading_row(dataset, gold)
        if grading is None:
            stats["grade_fail"] += 1
            continue
        try:
            value, extracted = bench.score(dataset, grading, pred)
        except Exception:
            stats["grade_fail"] += 1
            continue
        if value <= 0:
            stats["zero"] += 1
        elif dataset == "mbpp":
            stats["exec_pass"] += 1
        elif value < 0.999:
            stats["partial"] += 1
        else:
            alts = [norm(a) for a in gold.split("|")]
            if norm(extracted) in alts:
                stats["exact"] += 1
            else:
                stats["equiv"] += 1
                if len(dumps) < 400:
                    dumps.append({"gold": gold, "extracted": str(extracted)[:200],
                                  "pred_tail": str(pred)[-260:]})
    return stats, dumps


def main():
    outdir = ROOT / "audits" / "top3_award_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for method in ("daao", "maas", "flowbank"):
        for dataset in DATASETS:
            if method == "flowbank":
                pairs = rows_flowbank(dataset)
            else:
                pairs = rows_maas(method, dataset)
            if not pairs:
                summary.append({"cell": f"{method}/{dataset}", "items": 0,
                                "note": "no rows found"})
                continue
            stats, dumps = audit(method, dataset, pairs)
            summary.append(stats)
            if dumps:
                (outdir / f"{method}_{dataset}_equiv.json").write_text(
                    json.dumps(dumps, ensure_ascii=False, indent=1), encoding="utf-8")
            print(json.dumps(stats, ensure_ascii=False), flush=True)
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
