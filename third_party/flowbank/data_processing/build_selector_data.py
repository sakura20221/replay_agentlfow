from pathlib import Path
"""Build QueryMatching selector data from a curated workflow portfolio.

This is the bridge between CuraFlow (which selects a portfolio of workflows and
their per-query scores) and QueryMatching (which trains a selector over them).
It embeds the queries and workflow descriptions and writes the selector dataset
that ``QueryMatching/query_matching.py`` and ``inference.py`` consume.

Two modes
---------
1. Train+test mode (the real pipeline). The portfolio is evaluated on TWO
   separate query sets — the training/validation queries (from DiverseFlow's
   per-round results) and the held-out TEST queries (from running the selected
   workflows on test data, e.g. ``DiverseFlow/run_test.py``). They become the
   ``train`` and ``test`` rows of the selector's ``split`` column, so during
   QueryMatching training you watch the selector's test-set metric each epoch.

       --train-scores  {label: [effect_per_TRAIN_query]}   (+ --train-costs/--train-queries)
       --test-scores   {label: [effect_per_TEST_query]}    (+ --test-costs/--test-queries)

   Both must list the SAME workflow labels. Output split = train then test.

2. Single-set demo mode. One ``--scores`` set, split positionally by
   ``--split-ratio`` (no separate test evaluation). Convenient for a quick
   offline run; not how the paper numbers are produced.

       --scores  {label: [effect_per_query]}   (+ --costs/--queries/--split-ratio)

Common inputs
-------------
--descriptions Optional JSON ``{workflow_label: description}``. Missing labels
               fall back to the label string itself.
Score/cost JSON use the SAME ``{label: [..]}`` format CuraFlow consumes via
--sources. Missing costs default to 0.0 (cost-blind selection); missing query
text falls back to placeholders (fine for a runnable demo).

Embeddings
----------
--embedding-backend random   Deterministic seeded vectors — NO API key, fully
                             offline. The pipeline runs end-to-end; numbers will
                             not match the paper (random query features).
--embedding-backend openai   Real ``text-embedding-3-small`` (or --embedding-model)
                             via the OpenAI API. Reads OPENAI_API_KEY (and optional
                             OPENAI_BASE_URL) from the environment. Use this to
                             match paper-quality features.

Output (--out-dir)
------------------
    selector_data.npz            compact dataset (default; consumed directly)
    workflow_descriptions.json  {label: {"feature": description}}
    workflow_embeddings.pkl     (num_workflows, dim) float
    config.yaml                 structural defaults (llm_num, split, seed, ...)

Example (train+test mode)
-------------------------
    python data_processing/build_selector_data.py \
        --train-scores portfolio_train_scores.json --train-queries train_queries.json \
        --test-scores  portfolio_test_scores.json  --test-queries  test_queries.json \
        --task-id MATH --embedding-backend openai \
        --out-dir data/math_full
    python QueryMatching/query_matching.py --benchmark math_full --no_wandb
        # training_log.csv / console show test_result each epoch

Example (single-set demo)
-------------------------
    python data_processing/build_selector_data.py \
        --scores CuraFlow/examples/math_single_sources.json \
        --task-id MATH --embedding-backend random \
        --out-dir data/math_demo
    python QueryMatching/query_matching.py --benchmark math_demo --no_wandb
"""
import argparse
import hashlib
import json
import os
import pickle

import numpy as np
import yaml


def load_merged(paths):
    merged = {}
    n = None
    for p in paths:
        with open(p) as f:
            block = json.load(f)
        for label, vec in block.items():
            if label in merged:
                raise ValueError(f"duplicate label '{label}' across --scores")
            vec = [float(x) for x in vec]
            if n is None:
                n = len(vec)
            elif len(vec) != n:
                raise ValueError(f"'{label}' has {len(vec)} queries, expected {n}")
            merged[label] = vec
    if not merged:
        raise ValueError("no workflows loaded from --scores")
    return merged, n


def embed_random(texts, dim, salt):
    """Deterministic pseudo-embeddings: stable across runs, no API needed."""
    out = np.empty((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        h = hashlib.sha256((salt + "||" + str(t)).encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        out[i] = rng.standard_normal(dim)
    return out


def embed_minilm(texts, model_path):
    """Local sentence-transformers embeddings -- see shims/diverseflow/install.py.

    Chosen so that every method in the comparison shares one text encoder, and
    because no embedding API is reachable from this host. Deterministic: the
    encoder is frozen and inference-only.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device="cpu")
    vectors = model.encode([str(t) for t in texts], convert_to_numpy=True,
                           show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float64)


def resolve_embedding_creds(args, ap):
    """Resolve (base_url, api_key) for the OpenAI embedding client with precedence:
    explicit CLI flag > --config model entry > environment.

    --config points at a DiverseFlow-style YAML with a ``models:`` map. The entry
    read is ``--config-model`` (default: ``--embedding-model``); since the
    embedding model name (e.g. text-embedding-3-small) is usually not its own
    config entry, pass e.g. ``--config-model gpt-4o-mini`` to BORROW that model's
    OpenAI key + base_url for embeddings.
    """
    cfg_model = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        models = cfg.get("models", {}) or {}
        name = args.config_model or args.embedding_model
        if name not in models:
            ap.error(f"--config-model {name!r} not found in {args.config} "
                     f"(models: {', '.join(models) or 'none'}). Add an entry or pass "
                     f"--config-model to borrow another model's credentials.")
        cfg_model = models[name] or {}
    base_url = args.base_url or cfg_model.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or cfg_model.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        ap.error("no OpenAI API key for embeddings: set --api-key, a --config entry, "
                 "or $OPENAI_API_KEY.")
    return base_url, api_key


def embed_openai(texts, model, base_url, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    out = []
    B = 256
    for i in range(0, len(texts), B):
        batch = [str(t) for t in texts[i:i + B]]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float64)


def _load_costs(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def _load_queries(path, n, tag):
    if path:
        with open(path) as f:
            q = [str(x) for x in json.load(f)]
        if len(q) != n:
            raise ValueError(f"--{tag}-queries has {len(q)} items, expected {n}")
        return q
    return [f"{tag}_query_{i}" for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mode 1: separate train-eval + test-eval sets (the real pipeline)
    ap.add_argument("--train-scores", nargs="+")
    ap.add_argument("--test-scores", nargs="+")
    ap.add_argument("--train-costs")
    ap.add_argument("--test-costs")
    ap.add_argument("--train-queries")
    ap.add_argument("--test-queries")
    # Mode 2: single set, positional split (quick demo)
    ap.add_argument("--scores", nargs="+")
    ap.add_argument("--costs")
    ap.add_argument("--queries")
    ap.add_argument("--split-ratio", type=float, nargs=3, default=[0.2, 0.0, 0.8],
                    metavar=("TRAIN", "VAL", "TEST"))
    # Common
    ap.add_argument("--descriptions")
    ap.add_argument("--task-id", default="TASK")
    ap.add_argument("--task-description", default="")
    ap.add_argument("--metric", default="score")
    ap.add_argument("--seed", type=int, default=1881)
    ap.add_argument("--embedding-backend", choices=["random", "openai", "minilm"],
                    default="random")
    ap.add_argument("--minilm-path",
                    default=str(Path(__file__).resolve().parents[2]
                                / "shared" / "models" / "all-MiniLM-L6-v2"),
                    help="local sentence-transformers model for --embedding-backend minilm")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--embedding-dim", type=int, default=1536,
                    help="dimension for the random backend (openai uses model dim)")
    # OpenAI embedding credentials (precedence: CLI > --config entry > env)
    ap.add_argument("--config",
                    help="DiverseFlow-style YAML with a 'models:' map; supplies the "
                         "embedding base_url/api_key (e.g. DiverseFlow/config/config.yaml).")
    ap.add_argument("--config-model",
                    help="which --config entry to read base_url/api_key from "
                         "(default: --embedding-model). Use e.g. gpt-4o-mini to borrow "
                         "its OpenAI key for embeddings.")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL for embeddings (overrides --config; "
                         "falls back to $OPENAI_BASE_URL).")
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key for embeddings (overrides --config; "
                         "falls back to $OPENAI_API_KEY).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.train_scores and args.test_scores:
        # Train+test mode: concatenate the two evaluations into train/test rows.
        tr_scores, n_tr = load_merged(args.train_scores)
        te_scores, n_te = load_merged(args.test_scores)
        if set(tr_scores) != set(te_scores):
            raise ValueError("--train-scores and --test-scores must list the same "
                             f"workflow labels (got {set(tr_scores)} vs {set(te_scores)})")
        labels = list(tr_scores.keys())
        tr_costs, te_costs = _load_costs(args.train_costs), _load_costs(args.test_costs)
        queries = _load_queries(args.train_queries, n_tr, "train") + \
                  _load_queries(args.test_queries, n_te, "test")
        num_query = n_tr + n_te
        split_by_query = ["train"] * n_tr + ["test"] * n_te
        scores = {l: list(tr_scores[l]) + list(te_scores[l]) for l in labels}
        if tr_costs or te_costs:
            costs = {l: list(tr_costs.get(l, [0.0] * n_tr)) + list(te_costs.get(l, [0.0] * n_te))
                     for l in labels}
        else:
            costs = {}
        split_ratio = [round(n_tr / num_query, 4), 0.0, round(n_te / num_query, 4)]
    elif args.scores:
        # Single-set demo mode: positional split (no separate test evaluation).
        scores, num_query = load_merged(args.scores)
        labels = list(scores.keys())
        costs = _load_costs(args.costs)
        queries = _load_queries(args.queries, num_query, "query")
        tr = int(num_query * args.split_ratio[0])
        va = int(num_query * args.split_ratio[1])
        split_by_query = ["train"] * tr + ["val"] * va + ["test"] * (num_query - tr - va)
        split_ratio = list(args.split_ratio)
    else:
        ap.error("provide either --scores (single-set demo) OR both "
                 "--train-scores and --test-scores (train+test mode)")

    num_llms = len(labels)

    descriptions = {}
    if args.descriptions:
        with open(args.descriptions) as f:
            descriptions = json.load(f)
    desc_text = [
        (descriptions.get(l, {}).get("feature") if isinstance(descriptions.get(l), dict)
         else descriptions.get(l, l)) or l
        for l in labels
    ]

    # Embeddings
    if args.embedding_backend == "minilm":
        q_emb = embed_minilm(queries, args.minilm_path)
        w_emb = embed_minilm(desc_text, args.minilm_path)
        t_emb = embed_minilm([args.task_description or args.task_id], args.minilm_path)[0]
    elif args.embedding_backend == "random":
        q_emb = embed_random(queries, args.embedding_dim, "query")
        w_emb = embed_random(desc_text, args.embedding_dim, "workflow")
        t_emb = embed_random([args.task_description or args.task_id], args.embedding_dim, "task")[0]
    else:
        base_url, api_key = resolve_embedding_creds(args, ap)
        q_emb = embed_openai(queries, args.embedding_model, base_url, api_key)
        w_emb = embed_openai(desc_text, args.embedding_model, base_url, api_key)
        t_emb = embed_openai([args.task_description or args.task_id], args.embedding_model,
                             base_url, api_key)[0]

    # Row-major (query-major, workflow-minor) arrays
    effect, cost, raw_cost, split, llm, task_id = [], [], [], [], [], []
    for qi in range(num_query):
        for l in labels:
            effect.append(scores[l][qi])
            c = float(costs.get(l, [0.0] * num_query)[qi]) if costs else 0.0
            cost.append(c)
            raw_cost.append(c)
            split.append(split_by_query[qi])
            llm.append(l)
            task_id.append(args.task_id)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "selector_data.npz"),
        query_emb=q_emb, task_emb=t_emb,
        effect=np.asarray(effect, dtype=np.float64),
        cost=np.asarray(cost, dtype=np.float64),
        raw_cost=np.asarray(raw_cost, dtype=np.float64),
        split=np.asarray(split).astype("U"),
        llm=np.asarray(llm).astype("U"),
        task_id=np.asarray(task_id).astype("U"),
        metric=np.asarray([args.metric] * (num_query * num_llms)).astype("U"),
        query_text=np.asarray(queries).astype("U"),
        num_llms=np.int64(num_llms), num_query=np.int64(num_query),
    )
    with open(os.path.join(args.out_dir, "workflow_descriptions.json"), "w") as f:
        json.dump({l: {"feature": d} for l, d in zip(labels, desc_text)}, f, indent=2)
    with open(os.path.join(args.out_dir, "workflow_embeddings.pkl"), "wb") as f:
        pickle.dump(w_emb, f)
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({
            "llm_num": num_llms, "num_task": 1, "seed": args.seed,
            "split_ratio": split_ratio, "edge_dim": 3,
            "embedding_dim": 8,
            "learning_rate": 0.0003, "loss_type": "bce",
            "cost_weight": 0.0, "train_epoch": 500, "batch_size": 32,
            "train_mask_rate": 0.5, "weight_decay": 0.0001,
        }, f, default_flow_style=False)

    print(f"Wrote selector data for {num_llms} workflows x {num_query} queries -> {args.out_dir}/")
    print(f"  backend={args.embedding_backend}  split: "
          f"train={split_by_query.count('train')} val={split_by_query.count('val')} "
          f"test={split_by_query.count('test')}")
    print(f"  next: python QueryMatching/query_matching.py "
          f"--benchmark {os.path.basename(os.path.normpath(args.out_dir))} --no_wandb")


if __name__ == "__main__":
    main()
