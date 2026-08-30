#!/usr/bin/env python3
"""Drive FlowBank's stages 2 and 3 after DiverseFlow has produced a round pool.

FlowBank is precompute-and-reuse: stage 1 searches a pool of workflows, stage 2
(CuraFlow) selects a portfolio of k of them by oracle coverage, and stage 3
(QueryMatching) trains a selector that picks one workflow per query. Only stage 1
fits the (method, dataset, repeat) matrix in sweep.py -- stages 2 and 3 consume the
whole pool at once -- so they are driven from here.

The chain, following the repo's README:

  2a  data_processing/aggregate_round_scores.py   (TRAIN per-query scores)
  2b  CuraFlow/k_coverage_selection.py            (pick the size-k portfolio)
  3a  DiverseFlow/run_test.py --split test        (score the portfolio on held-out)
  3b  data_processing/aggregate_round_scores.py   (TEST per-query scores)
  3c  data_processing/describe_workflows.py       (workflow descriptions)
  3d  data_processing/build_selector_data.py      (selector dataset + embeddings)
  3e  QueryMatching/query_matching.py             (train; logs the test metric)

Two things to know about the numbers this produces:

* Stage 3a is the expensive step -- k workflows x the whole test split. With
  --max-k 6 and our four test splits (3120 items) that is up to ~19k item
  evaluations, comparable to an entire extra method. It is intrinsic to the
  method, not overhead we added.
* The embeddings use --embedding-backend minilm, added by shims/diverseflow.
  FlowBank's own default is OpenAI text-embedding-3-small, unreachable here; the
  `random` backend the repo also offers is documented by its own author as "does
  not match the paper". MiniLM is what five of the other six methods already use.

    python flowbank_pipeline.py --dataset SHARED_MATH --max-k 6
    python flowbank_pipeline.py --dataset SHARED_MATH --stages 2a 2b   # partial
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLOWBANK = ROOT / "third_party" / "flowbank"
DIVERSE = FLOWBANK / "DiverseFlow"
PY = str(ROOT / "envs" / "maas" / "bin" / "python")
PYG = str(ROOT / "envs" / "pyg" / "bin" / "python")
DEFAULT_RUNS = Path(os.getenv("SWEEP_RUNS", str(ROOT / "runs")))
DEFAULT_RUN_TAG = os.getenv("SWEEP_TAG", DEFAULT_RUNS.name)
PROXY_ROOT = os.getenv("SWEEP_PROXY_ROOT", "http://127.0.0.1:18080")

STAGES = ("2a", "2b", "3a", "3b", "3c", "3d", "3e")


def run(command: str, cwd: Path, label: str, namespace: str) -> None:
    print(f"\n[{label}] {command}", flush=True)
    env = dict(os.environ)
    env["SHIM_NAMESPACE"] = namespace
    env["SHIM_BASE_URL"] = f"{PROXY_ROOT}/{namespace}/v1"
    env["BASE_URL"] = f"{PROXY_ROOT}/{namespace}/v1/chat/completions"
    env["URL"] = f"{PROXY_ROOT}/{namespace}/v1"
    completed = subprocess.run(shlex.split(command), cwd=cwd, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"[{label}] failed with rc={completed.returncode}")


def round_dirs(dataset: str, optimized_path: str) -> list[int]:
    results = DIVERSE / optimized_path / dataset / "workflows" / "results.json"
    if not results.exists():
        raise SystemExit(f"stage 1 has not run: {results} missing")
    data = json.loads(results.read_text(encoding="utf-8"))
    # A round with score 0.0 for every problem is a workflow that failed to
    # execute, not one that answered everything wrong; feeding it to the coverage
    # selection would waste a portfolio slot on a workflow that cannot run.
    usable = [int(r["round"]) for r in data
              if isinstance(r.get("score"), (int, float)) and r["score"] > 0]
    if not usable:
        raise SystemExit(f"{results} has no round with a non-zero score")
    return sorted(usable)


def workflow_args(dataset: str, rounds: list[int], test: bool,
                  optimized_path: str) -> str:
    base = DIVERSE / optimized_path / dataset / "workflows"
    parts = []
    for n in rounds:
        path = base / f"round_{n}"
        if test:
            path = path / "test_results"
        parts.append(f"--workflow Flow_{n} {path}")
    return " ".join(parts)


def selected_rounds(out_dir: Path, fallback: list[int], k: int) -> list[int]:
    """Read CuraFlow's size-k portfolio out of k_coverage.csv.

    The file is a curve, one row per k, whose `best_combo` column holds the chosen
    labels (`Flow_1; Flow_3; Flow_22`). `oracle_gain_per_query` on the same row is
    the marginal coverage the k-th workflow buys -- worth reporting, because it
    shows where the portfolio saturates and therefore whether --max-k was generous
    or binding.
    """
    import csv

    path = out_dir / "k_coverage.csv"
    if not path.exists():
        print(f"  {path} missing; falling back to the {k} best-scoring rounds")
        return fallback[:k]

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print(f"  {path} is empty; falling back to the {k} best-scoring rounds")
        return fallback[:k]

    wanted = [r for r in rows if int(float(r["k"])) == k] or [rows[-1]]
    row = wanted[0]
    labels = [x.strip() for x in row["best_combo"].split(";") if x.strip()]
    picked = [int(label.replace("Flow_", "")) for label in labels
              if label.replace("Flow_", "").isdigit()]
    gain = row.get("oracle_gain_per_query")
    coverage = row.get("oracle_per_query")
    print(f"  CuraFlow k={row['k']}: portfolio={picked} "
          f"oracle_per_query={coverage} marginal_gain={gain}")
    for other in rows:
        print(f"    k={other['k']:>2}  oracle={other['oracle_per_query']}  "
              f"gain={other['oracle_gain_per_query']}")
    return picked or fallback[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="e.g. SHARED_MATH")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--exec_model_name", default="qwen3-8b")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--optimized-path", default="workspace")
    args = parser.parse_args()

    ds = args.dataset
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir
    work = runs_dir / "flowbank_pipeline" / ds.lower() / f"repeat{args.repeat}"
    work.mkdir(parents=True, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_tag)
    namespace_root = f"{args.run_tag}/flowbank_pipeline/{ds.lower()}/r{args.repeat}"
    (work / "run.json").write_text(
        json.dumps({"run_tag": args.run_tag, "repeat": args.repeat,
                    "dataset": ds, "optimized_path": args.optimized_path},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    pool = round_dirs(ds, args.optimized_path)
    print(f"  stage-1 pool: {len(pool)} usable round(s) -> {pool}")

    train_dir, test_dir = work / "train", work / "test"
    cura_dir = work / "curaflow"
    picked = pool[: args.max_k]

    if "2a" in args.stages:
        run(f"{PYG} data_processing/aggregate_round_scores.py "
            f"{workflow_args(ds, pool, test=False, optimized_path=args.optimized_path)} "
            f"--out {train_dir}",
            FLOWBANK, "2a aggregate TRAIN", f"{namespace_root}/2a")

    if "2b" in args.stages:
        run(f"{PYG} CuraFlow/k_coverage_selection.py --dataset-name {ds} "
            f"--sources {train_dir / 'sources.json'} --output-dir {cura_dir} "
            f"--max-k {args.max_k}",
            FLOWBANK, "2b k-coverage selection", f"{namespace_root}/2b")

    picked = selected_rounds(cura_dir, pool, args.max_k)
    print(f"  portfolio = {picked}")

    if "3a" in args.stages:
        run(f"{PY} run_test.py --dataset {ds} --rounds {' '.join(map(str, picked))} "
            f"--split test --exec_model_name {args.exec_model_name} "
            f"--optimized_path {args.optimized_path}",
            DIVERSE, "3a evaluate portfolio on TEST", f"{namespace_root}/3a")

    if "3b" in args.stages:
        run(f"{PYG} data_processing/aggregate_round_scores.py "
            f"{workflow_args(ds, picked, test=True, optimized_path=args.optimized_path)} "
            f"--out {test_dir}",
            FLOWBANK, "3b aggregate TEST", f"{namespace_root}/3b")

    if "3c" in args.stages:
        # describe_workflows generates the workflow descriptions with an LLM, so it
        # needs credentials; --config points it at the same proxy entry the search
        # used, keeping the descriptions on the same backbone as everything else.
        run(f"{PYG} data_processing/describe_workflows.py "
            f"{workflow_args(ds, picked, test=False, optimized_path=args.optimized_path)} "
            f"--dataset {ds} "
            f"--config DiverseFlow/config/config.yaml --model {args.exec_model_name} "
            f"--out {work / 'descriptions.json'}",
            FLOWBANK, "3c describe workflows", f"{namespace_root}/3c")

    benchmark = f"{ds.lower()}_{safe_tag}_r{args.repeat}"
    if "3d" in args.stages:
        # 2a aggregates the WHOLE pool (CuraFlow needs the full menu to pick
        # from), but the selector trains on train/test matrices over the SAME
        # workflow set. Re-aggregate TRAIN over the chosen portfolio first --
        # without this build_selector_data dies on a label mismatch (11 pool
        # flows vs 6 picked; found 2026-08-25 on SHARED_MATH, killed 3d for
        # every dataset while 3b/3c kept succeeding).
        run(f"{PYG} data_processing/aggregate_round_scores.py "
            f"{workflow_args(ds, picked, test=False, optimized_path=args.optimized_path)} "
            f"--out {train_dir}",
            FLOWBANK, "3d-prep re-aggregate TRAIN over portfolio",
            f"{namespace_root}/3d_prep")
        run(f"{PYG} data_processing/build_selector_data.py "
            f"--train-scores {train_dir / 'sources.json'} "
            f"--train-queries {train_dir / 'queries.json'} "
            f"--train-costs {train_dir / 'costs.json'} "
            f"--test-scores {test_dir / 'sources.json'} "
            f"--test-queries {test_dir / 'queries.json'} "
            f"--test-costs {test_dir / 'costs.json'} "
            f"--descriptions {work / 'descriptions.json'} "
            f"--task-id {ds} --embedding-backend minilm "
            f"--minilm-path {ROOT / 'shared/models/all-MiniLM-L6-v2'} "
            # The key check runs before the backend dispatch, so even the local
            # minilm backend needs credentials -- borrow the qwen3-8b entry 3c
            # already uses (the default --config-model is OpenAI's embedding
            # model, absent from our config). Found 2026-08-25: 3d died first
            # on "no OpenAI API key", then on the missing config-model entry.
            f"--config DiverseFlow/config/config.yaml --config-model qwen3-8b "
            f"--out-dir data/{benchmark}",
            FLOWBANK, "3d build selector data", f"{namespace_root}/3d")

    if "3e" in args.stages:
        run(f"{PYG} QueryMatching/query_matching.py --benchmark {benchmark} --no_wandb",
            FLOWBANK, "3e train selector", f"{namespace_root}/3e")
        log = FLOWBANK / "experiments" / benchmark
        print(f"\n  selector test metric is logged per epoch under {log}")


if __name__ == "__main__":
    main()
