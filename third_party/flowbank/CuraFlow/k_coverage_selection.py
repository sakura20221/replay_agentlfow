#!/usr/bin/env python3
"""CuraFlow (Stage 2 of FlowBank): coverage-aware portfolio curation.

Self-contained implementation of the curation procedure described in
Section 3.2 of the paper. Given a candidate workflow pool produced by
DiverseFlow (Stage 1), this script:

    1. Loads per-query scores for every candidate workflow.
    2. For each portfolio size k in [min_k, max_k], enumerates all size-k
       subsets and selects the one maximizing oracle coverage
       (Eq. 4: Coverage(S) = mean_q max_{w in S} score[w][q]).
       Ties are broken by lower mean pairwise correlation between the
       members' per-query score vectors (Section 3.2 tiebreaker).
    3. Computes the true pool ceiling = mean_q max_w score[w][q].
    4. Reports the saturation cutoff: smallest k such that
       Coverage(S_k*) >= saturation_ratio * pool_ceiling
       (Eq. 5; default saturation_ratio = 0.96).

Input format
------------
``--sources`` accepts one or more JSON files. Each file is a dict mapping
``workflow_label -> list[float]``: the per-query score (in ``[0, 1]``) of
that workflow on every query in a fixed evaluation set. All workflow
score lists must have the same length (one entry per query). Multiple
source files are merged into a single pool.

Example:

    {
      "math_w1": [1.0, 0.0, 1.0, 1.0, 0.5, ...],
      "math_w2": [0.0, 1.0, 1.0, 0.5, 1.0, ...],
      ...
    }

Outputs
-------
- ``<output-dir>/k_coverage.csv``         per-k oracle and tiebreaker stats
- ``<output-dir>/k_coverage.png``         coverage curve plot
- ``<output-dir>/k_coverage_meta.json``   sidecar with the pool ceiling

Usage
-----
    python k_coverage_selection.py \\
        --dataset-name MATH \\
        --sources scores.json \\
        --output-dir ./out \\
        --min-k 1 --max-k 8 \\
        --saturation-ratio 0.96
"""
import argparse
import csv
import json
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_workflows(sources):
    """Merge one or more JSON sources into a single ``label -> [scores]`` dict.

    Each source file must be a JSON object whose values are equal-length
    lists of per-query scores. Duplicate labels across sources raise
    ``ValueError``.
    """
    merged = {}
    expected_len = None
    for path in sources:
        with open(path, "r", encoding="utf-8") as f:
            block = json.load(f)
        if not isinstance(block, dict):
            raise ValueError(f"{path}: expected JSON object, got {type(block).__name__}")
        for label, scores in block.items():
            if label in merged:
                raise ValueError(f"duplicate workflow label '{label}' in {path}")
            if not isinstance(scores, list):
                raise ValueError(f"{path}: '{label}' must map to a list of scores")
            if expected_len is None:
                expected_len = len(scores)
            elif len(scores) != expected_len:
                raise ValueError(
                    f"{path}: '{label}' has {len(scores)} scores, "
                    f"expected {expected_len}"
                )
            merged[label] = [float(s) for s in scores]
    if not merged:
        raise ValueError("No workflows loaded from --sources")
    return merged


# ---------------------------------------------------------------------------
# Combinatorial size-k selection (oracle objective + correlation tiebreaker)
# ---------------------------------------------------------------------------
def combinatorial_selection_oracle(workflows, min_k=1, max_k=8):
    """Enumerate size-k subsets and pick the one maximizing oracle coverage.

    Implements Eq. (4) of the paper, with the Section 3.2 tiebreaker:
    among subsets of equal coverage, prefer the one with the lowest mean
    pairwise correlation between members' per-query score vectors.

    Returns ``per_k_info``: a list of dicts (one per k) with keys
    ``{k, oracle, avg_oracle_per_query, avg_pairwise_correlation_tiebreaker,
    redundancy_report, n_evaluated, n_oracle_ties, best_combo}``.
    """
    labels = list(workflows.keys())
    n_wf = len(labels)
    score_matrix = np.array([workflows[l] for l in labels], dtype=np.float64)
    num_queries = score_matrix.shape[1]

    # Pairwise correlation between workflows' per-query score vectors.
    if n_wf >= 2:
        corr_matrix = np.corrcoef(score_matrix)
        corr_matrix = np.where(np.isnan(corr_matrix), 0.0, corr_matrix)
    else:
        corr_matrix = np.array([[1.0]])

    per_k_info = []
    upper_k = min(max_k, n_wf)
    for k in range(min_k, upper_k + 1):
        best_score = -np.inf
        best_corr = np.inf
        best_combo_idx = None
        n_evaluated = 0
        n_ties = 0
        for combo in combinations(range(n_wf), k):
            n_evaluated += 1
            sub = score_matrix[list(combo)]                       # (k, Q)
            avg_oracle = float(sub.max(axis=0).mean())            # mean_q max_w
            if k >= 2:
                idx = np.array(combo)
                pair_block = corr_matrix[np.ix_(idx, idx)]
                # average of off-diagonal entries
                offdiag = pair_block[np.triu_indices(k, k=1)]
                avg_corr = float(offdiag.mean()) if offdiag.size else 0.0
            else:
                avg_corr = 0.0
            if avg_oracle > best_score + 1e-12:
                best_score = avg_oracle
                best_corr = avg_corr
                best_combo_idx = combo
                n_ties = 0
            elif abs(avg_oracle - best_score) <= 1e-12:
                n_ties += 1
                if avg_corr < best_corr - 1e-12:
                    best_corr = avg_corr
                    best_combo_idx = combo

        best_labels = [labels[i] for i in best_combo_idx]
        # Redundancy report: % of queries where >= 2 members get the max
        sub = score_matrix[list(best_combo_idx)]
        max_per_q = sub.max(axis=0)
        n_at_max = (sub == max_per_q[None, :]).sum(axis=0)
        redundancy = float((n_at_max >= 2).mean() * 100.0)

        per_k_info.append({
            "k": k,
            "oracle": float(best_score * num_queries),
            "avg_oracle_per_query": float(best_score),
            "avg_pairwise_correlation_tiebreaker": float(best_corr if k >= 2 else 0.0),
            "redundancy_report": redundancy,
            "n_evaluated": n_evaluated,
            "n_oracle_ties": n_ties,
            "best_combo": list(best_labels),
        })
    return per_k_info


# ---------------------------------------------------------------------------
# CSV / sidecar I/O
# ---------------------------------------------------------------------------
def load_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    per_k_info = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                per_k_info.append({
                    "k": int(r["k"]),
                    "oracle": float(r["oracle_total"]),
                    "avg_oracle_per_query": float(r["oracle_per_query"]),
                    "avg_pairwise_correlation_tiebreaker": float(r["avg_pair_corr_tiebreaker"]),
                    "redundancy_report": float(r["redundancy_report"]),
                    "n_evaluated": int(r["n_evaluated"]),
                    "n_oracle_ties": int(r["n_oracle_ties"]),
                    "best_combo": r["best_combo"].split("; "),
                })
            except (KeyError, ValueError):
                return None
    return per_k_info if per_k_info else None


def save_csv(per_k_info, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "oracle_total", "oracle_per_query", "oracle_gain_per_query",
            "avg_pair_corr_tiebreaker", "redundancy_report",
            "n_evaluated", "n_oracle_ties", "best_combo",
        ])
        prev = 0.0
        for info in per_k_info:
            avg_oracle = info["avg_oracle_per_query"]
            gain = avg_oracle - prev
            w.writerow([
                info["k"],
                f"{info['oracle']:.4f}",
                f"{avg_oracle:.4f}",
                f"{gain:.4f}",
                f"{info['avg_pairwise_correlation_tiebreaker']:.4f}",
                f"{info['redundancy_report']:.2f}",
                info["n_evaluated"],
                info["n_oracle_ties"],
                "; ".join(info["best_combo"]),
            ])
            prev = avg_oracle


def _meta_path(csv_path):
    return (csv_path.replace(".csv", "_meta.json")
            if csv_path.endswith(".csv") else csv_path + "_meta.json")


def _load_meta(csv_path):
    p = _meta_path(csv_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_meta(csv_path, meta):
    with open(_meta_path(csv_path), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Saturation rule (Eq. 5) — the only stop criterion we ship.
# ---------------------------------------------------------------------------
def _compute_pool_ceiling(workflows):
    """True pool ceiling = mean over queries of max-over-all-candidates per query."""
    M = np.array(list(workflows.values()), dtype=np.float64)
    return float(M.max(axis=0).mean())


def _find_saturation_k(per_k_info, ratio, anchor):
    """Smallest k whose oracle reaches ``ratio * anchor``."""
    target = ratio * anchor
    for info in per_k_info:
        if info["avg_oracle_per_query"] >= target:
            return info["k"]
    return per_k_info[-1]["k"]


def report_saturation(per_k_info, ratio, dataset_name, pool_ceiling):
    """Print the saturation-rule selected portfolio (Eq. 5)."""
    cutoff = ratio * pool_ceiling
    chosen_k = _find_saturation_k(per_k_info, ratio, pool_ceiling)
    selected = next(i for i in per_k_info if i["k"] == chosen_k)
    print(f"\n[{dataset_name}] Saturation cutoff @ ratio={ratio} "
          f"(target = {ratio} x {pool_ceiling:.4f} = {cutoff:.4f}): k*={chosen_k}")
    print(f"  Coverage = {selected['avg_oracle_per_query']:.4f}  "
          f"(= {selected['avg_oracle_per_query']/pool_ceiling*100:.1f}% of pool ceiling)")
    print(f"  Curated portfolio:")
    for label in selected["best_combo"]:
        print(f"    - {label}")
    return chosen_k


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_k_curve(per_k_info, dataset_name, n_candidates, png_path,
                 saturation_ratio, sat_stop_k, pool_ceiling):
    """Coverage curve with pool ceiling, saturation cutoff, and chosen k*."""
    ks = [0] + [info["k"] for info in per_k_info]
    coverages = [0.0] + [info["avg_oracle_per_query"] for info in per_k_info]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(ks, coverages, marker="o", linewidth=2, color="steelblue",
            label="Coverage (per-query avg)")
    ax.axhline(pool_ceiling, color="black", linestyle="-.", linewidth=1,
               alpha=0.6, label=f"pool ceiling = {pool_ceiling:.4f}")
    ax.axhline(saturation_ratio * pool_ceiling, color="darkorange",
               linestyle="--", linewidth=1, alpha=0.7,
               label=f"sat cutoff = {saturation_ratio:g} x ceiling "
                     f"({saturation_ratio*pool_ceiling:.4f})")
    ax.axvline(sat_stop_k, color="darkorange", linestyle=":", linewidth=1.8,
               alpha=0.9, label=f"sat@{saturation_ratio:g} (k*={sat_stop_k})")

    ax.set_xlabel("Portfolio size k")
    ax.set_ylabel("Coverage (per-query avg)")
    ax.set_title(f"{dataset_name} k-Coverage curve\n"
                 f"(pool: {n_candidates} candidates)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    for k, c in zip(ks, coverages):
        ax.annotate(f"{c:.3f}", (k, c),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Coverage-aware portfolio curation (CuraFlow, Stage 2 of FlowBank)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name (used for plot title and logging).")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="One or more JSON files mapping "
                             "workflow_label -> [score_per_query, ...].")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for k_coverage.csv / .png / _meta.json.")
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--saturation-ratio", type=float, default=0.96,
                        help="Saturation ratio tau (Eq. 5). Smallest k whose "
                             "coverage reaches tau x pool_ceiling is selected. "
                             "Default 0.96.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-run combinatorial enumeration even if a cached "
                             "k_coverage.csv exists in --output-dir.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "k_coverage.csv")

    cached_per_k = None if args.force_recompute else load_csv(csv_path)
    cached_meta = None if args.force_recompute else _load_meta(csv_path)
    workflows = None

    if cached_per_k is not None and len(cached_per_k) >= (args.max_k - args.min_k + 1):
        per_k_info = cached_per_k
        # Recover candidate-pool size: prefer the cached sidecar,
        # fall back to n_evaluated@k=1 (which equals n_wf).
        if cached_meta and "n_candidates" in cached_meta:
            n_workflows = int(cached_meta["n_candidates"])
        else:
            n_workflows = next((info["n_evaluated"] for info in per_k_info
                                if info["k"] == 1), len(per_k_info[0]["best_combo"]))
        print(f"[{args.dataset_name}] using cached CSV -> {csv_path} "
              f"(pass --force-recompute to re-run)")
    else:
        workflows = load_workflows(args.sources)
        n_workflows = len(workflows)
        n_q = len(next(iter(workflows.values())))
        print(f"[{args.dataset_name}] loaded {n_workflows} workflows x {n_q} queries")

        per_k_info = combinatorial_selection_oracle(
            workflows, min_k=args.min_k, max_k=args.max_k,
        )
        save_csv(per_k_info, csv_path)
        print(f"[{args.dataset_name}] saved CSV -> {csv_path}")

    # Pool ceiling: prefer cached sidecar; otherwise compute from workflows.
    if cached_meta and "pool_ceiling" in cached_meta:
        pool_ceiling = float(cached_meta["pool_ceiling"])
        print(f"[{args.dataset_name}] using cached pool ceiling = {pool_ceiling:.4f}")
    else:
        if workflows is None:
            workflows = load_workflows(args.sources)
        pool_ceiling = _compute_pool_ceiling(workflows)
        _save_meta(csv_path, {
            "pool_ceiling": pool_ceiling,
            "n_candidates": len(workflows),
            "n_queries": len(next(iter(workflows.values()))),
        })
        print(f"[{args.dataset_name}] pool ceiling = {pool_ceiling:.4f}")

    sat_stop_k = _find_saturation_k(per_k_info, args.saturation_ratio, pool_ceiling)

    png_path = os.path.join(args.output_dir, "k_coverage.png")
    plot_k_curve(per_k_info, args.dataset_name, n_workflows, png_path,
                 saturation_ratio=args.saturation_ratio,
                 sat_stop_k=sat_stop_k,
                 pool_ceiling=pool_ceiling)
    print(f"[{args.dataset_name}] saved plot -> {png_path}")

    report_saturation(per_k_info, args.saturation_ratio,
                      args.dataset_name, pool_ceiling)


if __name__ == "__main__":
    main()
