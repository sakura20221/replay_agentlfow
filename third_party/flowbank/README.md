# FlowBank: Query-Adaptive Agentic Workflows Optimization through Precompute-and-Reuse

*Precompute a diverse bank of agentic workflows offline, curate a compact high-coverage portfolio, then match each query to the right workflow at inference time — query-adaptive quality with no per-query search.*

[![Project Page](https://img.shields.io/badge/Project%20Page-FlowBank-8dbb3c)](https://agentic-flowbank.github.io/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2606.11290) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org/)

---

## 📚 Overview

Agentic workflows — multi-step LLM pipelines with operators like ensembling, code execution,
and self-revision — can substantially outperform a single model call, but **no single fixed
workflow is best across a heterogeneous query distribution**, and searching for a good workflow
*per query* at inference time is prohibitively expensive.

**FlowBank** takes a *precompute-and-reuse* view. Offline, it builds a small **bank** of
complementary workflows and learns to route to them; online, each query is answered by one
workflow selected with a single cheap forward pass — no search, no extra LLM calls for selecting.
The pipeline has three stages:

| Stage | Folder | What it does |
|-------|--------|--------------|
| **DiverseFlow** | [`DiverseFlow/`](DiverseFlow/) | Generates a *diverse pool* of candidate workflows via a two-phase MCTS-style search (performance warm-up → complementarity expansion). |
| **CuraFlow** | [`CuraFlow/`](CuraFlow/) | Curates a *compact, high-coverage portfolio* from the pool by size-`k` oracle-coverage selection with a correlation tiebreaker. |
| **QueryMatching** | [`QueryMatching/`](QueryMatching/) | Trains a graph-based *query-adaptive selector* that picks one portfolio workflow per query, `argmax_w f_θ(q, w)`. |

This repository provides two entry points:

1. [**Inference with existing examples**](#inference-with-existing-examples) — run the shipped trained selectors. Fast, deterministic, no API keys.
2. [**Run the full pipeline**](#run-the-full-pipeline) — DiverseFlow → CuraFlow → QueryMatching on the provided datasets. 

---

## 📦 Repository layout

```
FlowBank/
├── inference.py                 # reproduce the reported numbers from shipped selectors
├── requirements.txt             # deps for QueryMatching + CuraFlow + inference
├── data/                        # selector data, in two namespaces:
│   ├── example/<benchmark>/     #   shipped inference examples (selector_data.npz, workflow_*, config.yaml)
│   └── <benchmark>/             #   full-pipeline outputs you generate (build_selector_data.py --out-dir)
├── experiments/                 # trained selectors, same two namespaces:
│   ├── example/<benchmark>/<run>/   #   shipped reference runs (best_model.pth, test_predictions.json, ...)
│   └── <benchmark>/<run>/           #   your training outputs (query_matching.py)
├── datasets/                    # DiverseFlow eval splits (*_test.jsonl / *_validate.jsonl)
├── DiverseFlow/                 # Stage 1 (own requirements.txt + config)
├── CuraFlow/                    # Stage 2
│   ├── k_coverage_selection.py
│   └── examples/                # ready-to-run input (real 31-candidate MATH pool) + reference output
├── QueryMatching/               # Stage 3
│   ├── query_matching.py        #   training entry
│   └── model/{graph_nn.py, multi_task_graph_selector.py, data_io.py}
├── data_processing/             # bridge: DiverseFlow scores → CuraFlow → selector data
│   ├── aggregate_round_scores.py
│   └── build_selector_data.py
└── scripts/make_slim_data.py    # selector_data.csv → compact selector_data.npz
```

Four benchmarks ship with trained selectors: `drop`, `math`, `amc`, `mbpp` (DROP k=2, MATH k=4,
AMC k=5, MBPP+ScoreFlow k=2). The `example/` namespace holds these shipped artifacts; the full
pipeline writes to `data/<b>` and `experiments/<b>`, so your runs never overwrite an example.

---

## 🔧 Environments

Use two conda environments — one for the PyTorch stages, one for DiverseFlow.

**QueryMatching / CuraFlow / inference** (PyTorch + PyTorch-Geometric):

```bash
conda create -n flowbank python=3.10 -y && conda activate flowbank
pip install torch==2.1.0            # use the cuXXX wheel matching your GPU, or a CPU build
pip install -r requirements.txt
```

**DiverseFlow** (LLM client; no torch):

```bash
conda create -n diverseflow python=3.10 -y && conda activate diverseflow
pip install -r DiverseFlow/requirements.txt
```

---

## Inference with existing examples

```bash
conda activate flowbank
python inference.py --all            # all four benchmarks
python inference.py --all --assert   # non-zero exit if any |recomputed − stored| > 1e-4
python inference.py --benchmark drop # a single benchmark
```

Each run loads its `best_model.pth` and reports `result_predict` (the selector's mean test-set
score) against the value stored in `test_predictions.json`.

---

## Run the full pipeline

Each stage runs independently, and the shipped intermediate artifacts let you start at any
stage. Stages that need an LLM / embedding API are flagged.

### Stage 1 — DiverseFlow: generate workflows · needs an LLM API

```bash
conda activate diverseflow
cd DiverseFlow
cp config/config.example.yaml config/config.yaml   # then fill base_url + api_key
python run.py --dataset MATH --max_rounds 30 \
    --opt_model_name "<your-optimizer-model>" \
    --exec_model_name "gpt-4o-mini"
```

Outputs per-round workflows + per-query **train/validation** scores under
`DiverseFlow/workspace/<DATASET>/workflows/round_*/`.

### Stage 2 — CuraFlow: select a portfolio · offline

Aggregate the DiverseFlow round scores into `{label: [scores]}`, then run size-`k` selection:

```bash
# (a) DiverseFlow rounds → CuraFlow sources (one --workflow LABEL PATH per workflow)
python data_processing/aggregate_round_scores.py \
    --workflow DF_R8  DiverseFlow/workspace/MATH/workflows/round_8 \
    --workflow DF_R11 DiverseFlow/workspace/MATH/workflows/round_11 \
    --out runs/math_from_diverseflow

# (b) size-k oracle-coverage selection
python CuraFlow/k_coverage_selection.py --dataset-name MATH \
    --sources CuraFlow/examples/math_single_sources.json \
    --output-dir runs/curaflow_math --max-k 6
```

Step (b) runs as-is on the included example (`CuraFlow/examples/`, a real 31-candidate MATH
pool); a reference curve + CSV are in `CuraFlow/examples/math_single_output/`.

### Stage 3 — QueryMatching: train the selector

The selector trains on per-query scores for the portfolio over **both** train and test queries
(the test metric is logged each epoch). DiverseFlow produced only train scores, so first run the
selected workflows on the test split, then aggregate, describe, build, and train:

```bash
# (a) evaluate the CuraFlow-selected rounds on TEST data · needs an LLM API
#     writes per-test-query scores under workspace/<DS>/workflows/round_*/test_results/
cd DiverseFlow && python run_test.py --dataset MATH --rounds 8 11 17 \
    --exec_model_name gpt-4o-mini && cd ..

# (b) aggregate per-workflow scores → {label: [scores]}, once for TRAIN, once for TEST.
#     TRAIN = generation-time per-query evals written directly in each round_N/;
#     TEST  = what run_test.py wrote to round_N/test_results/. Use the SAME labels.
python data_processing/aggregate_round_scores.py \
    --workflow Flow_8  DiverseFlow/workspace/MATH/workflows/round_8 \
    --workflow Flow_11 DiverseFlow/workspace/MATH/workflows/round_11 \
    --workflow Flow_17 DiverseFlow/workspace/MATH/workflows/round_17 \
    --out runs/math_train
python data_processing/aggregate_round_scores.py \
    --workflow Flow_8  DiverseFlow/workspace/MATH/workflows/round_8/test_results \
    --workflow Flow_11 DiverseFlow/workspace/MATH/workflows/round_11/test_results \
    --workflow Flow_17 DiverseFlow/workspace/MATH/workflows/round_17/test_results \
    --out runs/math_test

# (c) describe each selected workflow: LLM-summarize its graph.py+prompt.py into a
#     paragraph, so the selector's workflow-node features are meaningful (not just the
#     label). · needs an LLM API (OPENAI_API_KEY). Optional: skip to fall back to labels.
python data_processing/describe_workflows.py --dataset MATH \
    --workflow Flow_8  DiverseFlow/workspace/MATH/workflows/round_8 \
    --workflow Flow_11 DiverseFlow/workspace/MATH/workflows/round_11 \
    --workflow Flow_17 DiverseFlow/workspace/MATH/workflows/round_17 \
    --out runs/math_descriptions.json

# (d) build the selector dataset from TRAIN + TEST scores (+ costs) + the descriptions.
#     Pass --*-costs or the cost column is all zero (see Common pitfalls).
#     --embedding-backend openai uses real text-embedding-3-small; supply the key via
#     $OPENAI_API_KEY, or reuse config.yaml with
#     --config DiverseFlow/config/config.yaml --config-model gpt-4o-mini
python data_processing/build_selector_data.py \
    --train-scores runs/math_train/sources.json --train-queries runs/math_train/queries.json \
    --train-costs  runs/math_train/costs.json \
    --test-scores  runs/math_test/sources.json  --test-queries  runs/math_test/queries.json \
    --test-costs   runs/math_test/costs.json \
    --descriptions runs/math_descriptions.json \
    --task-id MATH --embedding-backend openai --out-dir data/math_full

# (e) train; console + training_log.csv report the selector's test metric each epoch
python QueryMatching/query_matching.py --benchmark math_full --no_wandb
```

Sweepable parameters include `learning_rate`, `embedding_dim`, and `cost_weight` (the effect↔cost
blend), set via `--override`:

```bash
python QueryMatching/query_matching.py --benchmark math --no_wandb \
    --override learning_rate=1e-3 embedding_dim=16 cost_weight=0.1
```

To sweep a grid of these in parallel, use `run_sweep.py` (`--parallel N` runs at once). Each
combination trains into its own `experiments/<benchmark>/<run_name>/`, finished combos are skipped
on re-run, and a leaderboard + `experiments/<benchmark>/sweep_results.csv` summarize the grid:

```bash
python QueryMatching/run_sweep.py --benchmark math_full --parallel 4 \
    --learning-rate 1e-4 3e-4 1e-3 --embedding-dim 8 16 32 64 --cost-weight 0.0 0.1
```
---

## 📄 Citation
If you find our paper/code helpful, please kindly consider citing this work with the following reference:
```bibtex
@article{yuan2026flowbank,
  title={FlowBank: Query-Adaptive Agentic Workflows Optimization through Precompute-and-Reuse},
  author={Yuan, Lingzhi and Deng, Chenghao and Yu, Fangxu and Chakraborty, Souradip and Rostami, Mohammad and Huang, Furong},
  journal={arXiv preprint arXiv:2606.11290},
  year={2026}
}
```

## ✨ Acknowledgement
This work is based on the amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [AFlow](https://github.com/FoundationAgents/AFlow)
- [ScoreFlow](https://github.com/Gen-Verse/ScoreFlow)
- [GraphRouter](https://github.com/ulab-uiuc/GraphRouter)