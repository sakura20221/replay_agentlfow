"""Inference with existing examples: run the shipped FlowBank selectors on test data.

This loads a trained query-adaptive selector (``best_model.pth``) together with
its selector data and computes the test-set metric ``result_predict`` (the mean
score of the workflow the selector picks per test query), then compares it to the
value stored in that run's ``test_predictions.json``.

Usage
-----
    python inference.py --benchmark drop        # one of: drop, math, amc, mbpp
    python inference.py --all                    # all four, with a summary table
    python inference.py --all --assert           # non-zero exit if any |delta| > tol
    python inference.py --run-config experiments/<...>/config.yaml   # arbitrary run

Environment
-----------
Use the project PyTorch env from the README (torch 2.1.0+cu121, torch_geometric 2.7.0).
The eval path is deterministic (no dropout; eval()+no_grad(); BatchNorm uses the
checkpoint's running stats; the train/test split is read from the data's
materialized ``split`` column), so results match to floating-point tolerance.
"""
import argparse
import json
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
QM_DIR = os.path.join(REPO_ROOT, "QueryMatching")
sys.path.insert(0, os.path.join(QM_DIR, "model"))
sys.path.insert(0, QM_DIR)

from model.multi_task_graph_selector import graph_selector_prediction  # noqa: E402


# benchmark name (= dir under data/ and experiments/) -> run name, for the four
# results reported in the paper.
BENCHMARKS = {
    "drop": "lr3e-04_emb32_cw0.0",
    "math": "lr1e-03_emb16_cw0.1",
    "amc":  "lr1e-03_emb8_cw0.3",
    "mbpp": "lr3e-04_emb64_cw0.1",
}


class _DummyWandb:
    def log(self, *a, **kw):
        pass

    def init(self, *a, **kw):
        pass


def _run_dir_for(benchmark):
    # Shipped inference examples live under the example/ namespace, separate from
    # full-pipeline outputs (which go to data/<b> and experiments/<b>).
    return os.path.join(REPO_ROOT, "experiments", "example", benchmark, BENCHMARKS[benchmark])


def _localize_config(config, run_dir):
    """Rewrite path fields so a shipped config works after cloning anywhere."""
    bench = config.get("benchmark") or _infer_bench_dir(config)
    bench = os.path.basename(os.path.normpath(bench))
    data_dir = os.path.join(REPO_ROOT, "data", "example", bench)
    config["saved_selector_data_path"] = os.path.join(data_dir, "selector_data.npz")
    config["llm_description_path"] = os.path.join(data_dir, "workflow_descriptions.json")
    config["llm_embedding_path"] = os.path.join(data_dir, "workflow_embeddings.pkl")
    config["model_path"] = os.path.join(run_dir, "best_model.pth")
    config["test_predictions_path"] = os.path.join(run_dir, "test_predictions.json")
    config["train_log_path"] = os.path.join(run_dir, "training_log.csv")
    config.setdefault("gnn_type", "default")
    return config


def _infer_bench_dir(config):
    # Fallback when a config has no explicit `benchmark` key: derive it from the
    # data path (.../data/<bench>/selector_data.npz).
    p = config.get("saved_selector_data_path", "")
    return os.path.basename(os.path.dirname(p)) if p else ""


def run_inference(run_dir, save_predictions=False):
    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config = _localize_config(config, run_dir)

    selector = graph_selector_prediction(
        selector_data_path=config["saved_selector_data_path"],
        llm_path=config["llm_description_path"],
        llm_embedding_path=config["llm_embedding_path"],
        config=config,
        wandb=_DummyWandb(),
        eval_only=True,
        save_predictions=save_predictions,
    )
    recomputed = selector.result_predict

    with open(os.path.join(run_dir, "test_predictions.json")) as f:
        stored = json.load(f)["summary"]["result_predict"]

    return {
        "run_dir": os.path.relpath(run_dir, REPO_ROOT),
        "recomputed": recomputed,
        "stored": stored,
        "golden": selector.result_golden,
        "delta": abs(recomputed - stored),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--benchmark", choices=list(BENCHMARKS), help="run inference on one benchmark")
    g.add_argument("--all", action="store_true", help="run inference on all four benchmarks")
    g.add_argument("--run-config", help="path to a run dir's config.yaml")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="exit non-zero if any |recomputed - stored| > --tol")
    ap.add_argument("--tol", type=float, default=1e-4, help="tolerance for --assert")
    ap.add_argument("--save-predictions", action="store_true",
                    help="re-write test_predictions.json for the run")
    args = ap.parse_args()

    if args.run_config:
        run_dirs = [os.path.dirname(os.path.abspath(args.run_config))]
    elif args.all:
        run_dirs = [_run_dir_for(b) for b in BENCHMARKS]
    else:
        run_dirs = [_run_dir_for(args.benchmark)]

    results = [run_inference(rd, save_predictions=args.save_predictions) for rd in run_dirs]

    print("\n" + "=" * 78)
    print(f"{'run':<52}{'recomputed':>9}{'stored':>9}{'|d|':>9}")
    print("-" * 78)
    worst = 0.0
    for r in results:
        worst = max(worst, r["delta"])
        print(f"{r['run_dir']:<52}{r['recomputed']:>9.4f}{r['stored']:>9.4f}{r['delta']:>9.1e}")
    print("=" * 78)
    print(f"max |delta| = {worst:.2e}  (tol = {args.tol:.0e})")

    if args.do_assert and worst > args.tol:
        print("FAIL: inference result exceeded tolerance vs stored.")
        sys.exit(1)
    print("OK" if args.do_assert else "")


if __name__ == "__main__":
    main()
