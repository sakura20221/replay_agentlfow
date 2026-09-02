"""Parallel hyperparameter sweep for the QueryMatching selector (Stage 3).

Trains ``query_matching.py`` over a grid of (learning_rate, embedding_dim,
cost_weight), up to ``--parallel`` runs concurrently, then prints a leaderboard
and writes a CSV. Each combination trains into its own
``experiments/<benchmark>/<run_name>/`` (the run name encodes the swept
hyperparameters), so runs never collide and already-finished combos are skipped
on re-run — kill it and re-launch to resume.

Usage
-----
    # list the grid without running anything
    python QueryMatching/run_sweep.py --benchmark math_full --dry-run

    # default grid (lr x emb), 4 runs at a time
    python QueryMatching/run_sweep.py --benchmark math_full --parallel 4

    # custom grid, including a cost_weight axis
    python QueryMatching/run_sweep.py --benchmark math_full --parallel 4 \
        --learning-rate 3e-4 1e-3 --embedding-dim 8 16 32 --cost-weight 0.0 0.1

Each run goes through the normal ``query_matching.py`` entry, so its outputs
(best_model.pth, test_predictions.json, training_log.csv) land in the usual
per-run directory. The sweep summary CSV is written to
``experiments/<benchmark>/sweep_results.csv``.
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))       # .../QueryMatching
REPO_ROOT = os.path.dirname(HERE)
QM = os.path.join(HERE, "query_matching.py")

# Default sweep axes (the paper's grid). cost_weight is a single value by default;
# add more to sweep the effect<->cost blend.
DEFAULT_GRID = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "embedding_dim": [8, 16, 32, 64],
    "cost_weight": [0.0],
}


def run_name(p):
    # MUST match query_matching.derive_paths so we can locate each run's output dir.
    return "lr{:.0e}_emb{}_cw{}".format(
        p["learning_rate"], p["embedding_dim"], p["cost_weight"])


def out_dir(benchmark, p):
    return os.path.join(REPO_ROOT, "experiments", benchmark, run_name(p))


def train_one(benchmark, p, train_epoch, timeout):
    """Run one combo via query_matching.py. Returns (ok, stderr)."""
    overrides = [f"learning_rate={p['learning_rate']}",
                 f"embedding_dim={p['embedding_dim']}",
                 f"cost_weight={p['cost_weight']}"]
    if train_epoch is not None:
        overrides.append(f"train_epoch={train_epoch}")
    cmd = [sys.executable, QM, "--benchmark", benchmark, "--no_wandb",
           "--override", *overrides]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stderr


def collect(benchmark, p):
    """Read a completed run's summary, or None if it didn't produce one."""
    pred = os.path.join(out_dir(benchmark, p), "test_predictions.json")
    if not os.path.exists(pred):
        return None
    with open(pred) as f:
        s = json.load(f)["summary"]
    sel, orc = s.get("selector", {}), s.get("oracle", {})
    return {
        "run_name": run_name(p), **p,
        "result_predict": s["result_predict"], "result_golden": s["result_golden"],
        "gap": s["result_golden"] - s["result_predict"],
        "selector_effect": sel.get("effect", ""), "selector_cost": sel.get("cost", ""),
        "oracle_effect": orc.get("effect", ""), "oracle_cost": orc.get("cost", ""),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="benchmark = directory under data/")
    ap.add_argument("--learning-rate", type=float, nargs="+",
                    default=DEFAULT_GRID["learning_rate"])
    ap.add_argument("--embedding-dim", type=int, nargs="+",
                    default=DEFAULT_GRID["embedding_dim"])
    ap.add_argument("--cost-weight", type=float, nargs="+",
                    default=DEFAULT_GRID["cost_weight"])
    ap.add_argument("--parallel", type=int, default=1,
                    help="number of concurrent training runs (default 1)")
    ap.add_argument("--train-epoch", type=int, default=None,
                    help="override train_epoch for every run (else uses config default)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-run timeout in seconds (default 3600)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the grid and exit without training")
    args = ap.parse_args()

    grid = {"learning_rate": args.learning_rate,
            "embedding_dim": args.embedding_dim,
            "cost_weight": args.cost_weight}
    combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())]

    print(f"Sweep: {len(combos)} combinations  (parallel={args.parallel})")
    for k, v in grid.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        for i, p in enumerate(combos, 1):
            print(f"  {i:>3}. {run_name(p)}")
        print("(dry run — nothing executed)")
        return

    def process(p):
        if os.path.exists(os.path.join(out_dir(args.benchmark, p), "test_predictions.json")):
            return p, collect(args.benchmark, p), "skip"
        ok, err = train_one(args.benchmark, p, args.train_epoch, args.timeout)
        if ok:
            return p, collect(args.benchmark, p), "ok"
        return p, None, "fail:" + (err or "")[-300:]

    results, failed = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futs = {ex.submit(process, p): p for p in combos}
        for i, fut in enumerate(as_completed(futs), 1):
            p, res, status = fut.result()
            if status.startswith("fail"):
                failed += 1
                print(f"[{i}/{len(combos)}] {run_name(p)} — FAILED\n    {status[5:]}")
            elif res:
                results.append(res)
                tag = "skipped" if status == "skip" else "done"
                print(f"[{i}/{len(combos)}] {run_name(p)} — {tag}  "
                      f"predict={res['result_predict']:.4f} golden={res['result_golden']:.4f}")

    print(f"\nFinished in {time.time() - t0:.0f}s  ok={len(results)}  failed={failed}")
    if not results:
        return

    sweep_csv = os.path.join(REPO_ROOT, "experiments", args.benchmark, "sweep_results.csv")
    os.makedirs(os.path.dirname(sweep_csv), exist_ok=True)
    cols = ["run_name", "learning_rate", "embedding_dim", "cost_weight",
            "result_predict", "result_golden", "gap",
            "selector_effect", "selector_cost", "oracle_effect", "oracle_cost"]
    rows = sorted(results, key=lambda r: r["result_predict"], reverse=True)
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {sweep_csv}")

    print(f"\n{'rank':<5}{'run_name':<28}{'predict':>9}{'golden':>9}{'gap':>8}")
    print("-" * 59)
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['run_name']:<28}{r['result_predict']:>9.4f}"
              f"{r['result_golden']:>9.4f}{r['gap']:>8.4f}")


if __name__ == "__main__":
    main()
