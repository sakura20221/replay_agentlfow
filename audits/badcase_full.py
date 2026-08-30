"""Full bad-case census for daao / maas across all 5 datasets, with cause tagging.

For every wrong item, tag the failure cause by inspecting the full prediction:
  EMPTY        - nothing produced
  NO_FINAL     - produced text but no boxed/Answer marker
  LOOP         - degenerate repetition
  WRONG_VALUE  - gave a clean final answer, it is simply wrong
  PARTIAL_F1   - drop only, 0<score<1
Also records per-item solved flags so we can do overlap analysis.
"""
import csv, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SWEEP_RUNS", str(ROOT / "runs_v5"))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "shared"))
os.chdir(ROOT)
import bench
import importlib.util
spec = importlib.util.spec_from_file_location("c", ROOT / "collect.py")
collect = importlib.util.module_from_spec(spec); spec.loader.exec_module(collect)
csv.field_size_limit(sys.maxsize)

KEY = {"math": "SHARED_MATH", "amc": "SHARED_AMC", "drop": "SHARED_DROP",
       "mmlu_pro": "SHARED_MMLUPRO", "mbpp": "SHARED_MBPP"}
DATASETS = ["math", "amc", "mbpp", "drop", "mmlu_pro"]


def looping(text, min_len=400):
    if len(text) < min_len:
        return False
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    if len(lines) < 6:
        return False
    c = Counter(lines)
    return c.most_common(1)[0][1] >= 4


def tag(dataset, score, pred, extracted):
    if not (pred or "").strip():
        return "EMPTY"
    if dataset == "mbpp":
        return "CODE_FAIL"
    if looping(pred):
        return "LOOP"
    if not re.search(r"\\boxed\{|Answer:|answer is", pred, re.I):
        return "NO_FINAL"
    if dataset == "drop" and 0 < score < 1:
        return "PARTIAL_F1"
    return "WRONG_VALUE"


def cell(method, dataset):
    base = ROOT / "third_party" / method / method / "ext" / "maas" / "scripts" / "optimized" / KEY[dataset] / "test"
    job = ROOT / "runs_v5" / method / dataset / "repeat1"
    floor = 0.0
    for m in ("test.cmd", "search.cmd"):
        if (job / m).exists():
            floor = (job / m).stat().st_mtime; break
    files = [f for f in base.glob("round_*/0.*.csv") if f.stat().st_mtime >= floor]
    if not files:
        return {}, {}
    path = max(files, key=lambda p: p.stat().st_mtime)
    rows, causes = {}, Counter()
    with path.open(newline="", encoding="utf-8", errors="replace") as h:
        for row in csv.DictReader(h):
            gold = (row.get("expected_output") or "").strip()
            if not gold:
                continue
            g = collect._grading_row(dataset, gold)
            if g is None:
                continue
            pred = row.get("prediction") or ""
            try:
                v, ex = bench.score(dataset, g, pred)
            except Exception:
                continue
            q = str(row.get("question") or "")
            k = gold + "||" + q[:120]
            rows[k] = {"score": v, "gold": gold, "q": q[:300], "pred": pred,
                       "extracted": str(ex)[:100]}
            if v < 0.999:
                causes[tag(dataset, v, pred, ex)] += 1
    return rows, causes


def main():
    out = ROOT / "audits" / "badcase_full"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in DATASETS:
        a, ca = cell("daao", ds)
        b, cb = cell("maas", ds)
        keys = set(a) | set(b)
        both_wrong, only_daao, only_maas = [], [], []
        for k in keys:
            va = a.get(k, {}).get("score")
            vb = b.get(k, {}).get("score")
            if va is None or vb is None:
                continue
            oka, okb = va > 0.999, vb > 0.999
            if not oka and not okb:
                both_wrong.append(k)
            elif oka and not okb:
                only_daao.append(k)
            elif okb and not oka:
                only_maas.append(k)
        report[ds] = {
            "n": len(keys),
            "daao_causes": dict(ca), "maas_causes": dict(cb),
            "both_wrong": len(both_wrong),
            "daao_solved_maas_not": len(only_daao),
            "maas_solved_daao_not": len(only_maas),
        }
        print(f"{ds:10} n={len(keys):5} both_wrong={len(both_wrong):4} "
              f"daao_only={len(only_daao):3} maas_only={len(only_maas):3}", flush=True)
        print(f"           daao causes: {dict(ca)}")
        print(f"           maas causes: {dict(cb)}", flush=True)
        dump = {"both_wrong": [{"gold": a[k]["gold"], "q": a[k]["q"],
                                "daao_extracted": a[k]["extracted"],
                                "daao_tail": a[k]["pred"][-300:]}
                               for k in both_wrong[:80] if k in a],
                "split": [{"gold": (a.get(k) or b[k])["gold"],
                           "q": (a.get(k) or b[k])["q"],
                           "winner": "daao" if k in only_daao else "maas",
                           "daao_extracted": a.get(k, {}).get("extracted"),
                           "maas_extracted": b.get(k, {}).get("extracted")}
                          for k in (only_daao + only_maas)[:80]]}
        (out / f"{ds}.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
