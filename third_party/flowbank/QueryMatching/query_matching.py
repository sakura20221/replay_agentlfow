"""QueryMatching (Stage 3 of FlowBank).

Trains the graph-based query-adaptive selector over a curated workflow
portfolio: a heterogeneous 2-layer GNN encoder followed by an MLP decoder
that predicts each query-workflow edge value
``v_{q,w} = (1 - lambda) * effect + lambda * (1 - cost)`` (Eq. 6).
At inference time the selector picks ``argmax_w f_theta(q, w)`` (Section 3.3).

Usage
-----
    python query_matching.py --benchmark math
    python query_matching.py --benchmark math --override learning_rate=1e-3 embedding_dim=16

Input layout (read from data/{benchmark}/, falling back to the shipped
data/example/{benchmark}/ if the pipeline namespace is empty):

    {data_dir}/selector_data.npz       (or selector_data.csv)
    {data_dir}/workflow_descriptions.json
    {data_dir}/workflow_embeddings.pkl
    {data_dir}/config.yaml             (optional, overrides DEFAULT_CONFIG)

Outputs are always written to the pipeline namespace
``experiments/{benchmark}/lr{LR}_emb{DIM}_cw{CW}/`` (never experiments/example/).
"""
import argparse
import os
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model.multi_task_graph_selector import graph_selector_prediction


# data/ and experiments/ live at the repository root (shared with inference.py),
# which is the parent of this file's directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Default training hyperparameters (overridden by data/{benchmark}/config.yaml
# when present, then by --override key=value pairs).
DEFAULT_CONFIG = {
    "seed": 1881,
    "train_epoch": 500,
    "cost_weight": 0.0,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "train_mask_rate": 0.5,
    "batch_size": 32,
    "num_task": 1,
    "split_ratio": [0.2, 0.0, 0.8],
    "embedding_dim": 8,
    "edge_dim": 3,
    "loss_type": "bce",
}


def resolve_data_dir(benchmark):
    """Locate a benchmark's data dir: prefer the pipeline namespace data/<b>,
    fall back to the shipped inference example data/example/<b>."""
    d = os.path.join(PROJECT_ROOT, "data", benchmark)
    if os.path.isdir(d):
        return d
    ex = os.path.join(PROJECT_ROOT, "data", "example", benchmark)
    if os.path.isdir(ex):
        return ex
    return d  # non-existent; caller reports the error


def derive_paths(benchmark, config, data_dir):
    """Map a benchmark + hyperparameters to its data/output paths.

    Inputs come from ``data_dir`` (data/<b> or the shipped data/example/<b>);
    outputs ALWAYS go to the pipeline namespace experiments/<benchmark>/<run>/
    (never the example/ tree), where the run name encodes the swept
    hyperparameters, e.g. ``lr1e-03_emb16_cw0.1``.
    """
    run_name = "lr{:.0e}_emb{}_cw{}".format(
        config["learning_rate"], config["embedding_dim"], config["cost_weight"])
    exp_dir = os.path.join(PROJECT_ROOT, "experiments", benchmark, run_name)
    return {
        # data_io resolves this to selector_data.npz (shipped) or selector_data.csv
        # (pipeline-generated) — whichever exists in the data dir.
        "saved_selector_data_path": os.path.join(data_dir, "selector_data.npz"),
        "llm_description_path": os.path.join(data_dir, "workflow_descriptions.json"),
        "llm_embedding_path": os.path.join(data_dir, "workflow_embeddings.pkl"),
        "model_path": os.path.join(exp_dir, "best_model.pth"),
        "train_log_path": os.path.join(exp_dir, "training_log.csv"),
        "test_predictions_path": os.path.join(exp_dir, "test_predictions.json"),
    }


def load_config(args):
    """Build the training config from defaults, optional YAML, and CLI overrides."""
    data_dir = resolve_data_dir(args.benchmark)
    if not os.path.isdir(data_dir):
        available = []
        for root in (os.path.join(PROJECT_ROOT, "data"),
                     os.path.join(PROJECT_ROOT, "data", "example")):
            if os.path.isdir(root):
                available += [d for d in sorted(os.listdir(root))
                             if os.path.isdir(os.path.join(root, d)) and d != "example"]
        print(f"Error: data/{args.benchmark}/ (or data/example/{args.benchmark}/) not found.")
        if available:
            print(f"Available: {', '.join(sorted(set(available)))}")
        sys.exit(1)

    config = dict(DEFAULT_CONFIG)

    data_config_path = os.path.join(data_dir, "config.yaml")
    if os.path.exists(data_config_path):
        with open(data_config_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f) or {}
        config.update(data_config)
        print(f"Loaded config from {os.path.relpath(data_config_path, PROJECT_ROOT)}")

    if args.override:
        for kv in args.override:
            key, val = kv.split("=", 1)
            try:
                val = yaml.safe_load(val)
            except yaml.YAMLError:
                pass
            # YAML leaves unquoted scientific notation ("1e-3") as a string; coerce
            # numeric-looking values so e.g. learning_rate=1e-3 becomes a float.
            if isinstance(val, str):
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            config[key] = val

    paths = derive_paths(args.benchmark, config, data_dir)
    config.update(paths)
    config["gnn_type"] = args.gnn_type
    loss_type = config.get("loss_type", "bce")
    print(f"Benchmark: {args.benchmark}, GNN: {args.gnn_type}, Loss: {loss_type}")
    print(f"  Data:   {os.path.dirname(paths['saved_selector_data_path'])}/")
    print(f"  Output: {os.path.dirname(paths['model_path'])}/")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="QueryMatching: train the graph-based query-adaptive selector."
    )
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Benchmark name = directory under data/.")
    parser.add_argument("--gnn_type", type=str, default="default",
                        choices=["default"],
                        help="GNN architecture (only 'default' is shipped here).")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging.")
    parser.add_argument("--override", nargs="*",
                        help="Override config values, e.g. --override learning_rate=1e-3 "
                             "embedding_dim=16")
    args = parser.parse_args()

    config = load_config(args)

    for path_key in ["model_path", "train_log_path", "test_predictions_path"]:
        if path_key in config:
            os.makedirs(os.path.dirname(config[path_key]) or ".", exist_ok=True)

    if args.no_wandb or not config.get("wandb_key"):
        class _DummyWandb:
            def log(self, *a, **kw):
                pass

            def init(self, *a, **kw):
                pass

        wandb = _DummyWandb()
        print("Running without wandb logging.")
    else:
        import wandb
        wandb.login(key=config["wandb_key"])
        wandb.init(project="graph_selector")

    print("Using GNN architecture: default (with task node)")
    graph_selector_prediction(
        selector_data_path=config["saved_selector_data_path"],
        llm_path=config["llm_description_path"],
        llm_embedding_path=config["llm_embedding_path"],
        config=config,
        wandb=wandb,
    )


if __name__ == "__main__":
    main()
