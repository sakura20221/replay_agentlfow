"""Convert a verbose ``selector_data.csv`` into a compact, lossless ``selector_data.npz``.

Why this exists
---------------
The raw per-query selection table stores each 1536-d embedding as JSON *text*,
duplicated once per workflow row, with the (identical) task embedding repeated on
every single row. For the shipped benchmarks that makes each CSV 50-260 MB while
~99% of the bytes are redundant embedding text.

This script stores every embedding exactly once, as a binary float64 array (the
same dtype ``json.loads`` produces), shrinking the four shipped datasets from
~590 MB to ~35 MB total with **zero numerical change**: the selector casts
embeddings to ``torch.float`` regardless, so the float64 ``.npz`` arrays match
the original results bit-for-bit (see ``QueryMatching/model/data_io.py``).

The ``.npz`` keeps the original CSV *row order* (query-major, workflow-minor) and
the materialized ``split`` column, so the train/val/test partition is preserved.

Usage
-----
    python scripts/make_slim_data.py \
        --csv  /path/to/selector_data.csv \
        --out  data/<benchmark>/selector_data.npz \
        --num-llms 2                      # or --descriptions data/<b>/workflow_descriptions.json

The ``.npz`` is consumed transparently by ``QueryMatching`` (training) and
``inference.py`` (inference) via ``QueryMatching/model/data_io.py``.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

csv.field_size_limit(10 ** 9)


def _parse_embedding(s):
    """Parse a CSV embedding cell to a flat python list (matches the selector)."""
    s = s.strip()
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        import re
        s2 = re.sub(r"\s+", ", ", s)
        try:
            val = json.loads(s2)
        except json.JSONDecodeError:
            val = json.loads(s2.replace("[[,", "[["))
    # Stored as a nested list [[...]]; take the inner vector.
    return val[0] if (isinstance(val, list) and len(val) and isinstance(val[0], list)) else val


def convert(csv_path, out_path, num_llms):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        required = {"query_embedding", "task_description_embedding", "effect", "cost"}
        missing = required - set(cols)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")
        has_raw_cost = "raw_cost" in cols
        has_split = "split" in cols
        has_metric = "metric" in cols

        effect, cost, raw_cost = [], [], []
        split, llm, task_id, metric = [], [], [], []
        query_text, ground_truth = [], []
        query_emb = []            # one row per query (every num_llms-th CSV row)
        task_emb_first = None
        task_emb_const = True

        for i, row in enumerate(reader):
            effect.append(float(row["effect"]))
            cost.append(float(row["cost"]))
            raw_cost.append(float(row["raw_cost"]) if has_raw_cost else float(row["cost"]))
            split.append(row["split"] if has_split else "")
            llm.append(row.get("llm", ""))
            task_id.append(row.get("task_id", ""))
            metric.append(row.get("metric", "") if has_metric else "")
            query_text.append(row.get("query", ""))
            ground_truth.append(row.get("ground_truth", ""))
            # Dedup query embedding: first row of each query block.
            if i % num_llms == 0:
                query_emb.append(_parse_embedding(row["query_embedding"]))
            # Verify the task embedding is constant (single-task datasets).
            te = _parse_embedding(row["task_description_embedding"])
            if task_emb_first is None:
                task_emb_first = te
            elif task_emb_const and i % num_llms == 0:
                if not np.allclose(np.asarray(te), np.asarray(task_emb_first), atol=1e-9):
                    task_emb_const = False

    num_rows = len(effect)
    if num_rows % num_llms != 0:
        raise ValueError(f"row count {num_rows} not divisible by num_llms {num_llms}")
    num_query = num_rows // num_llms

    query_emb = np.asarray(query_emb, dtype=np.float64)        # (num_query, dim)
    task_emb = np.asarray(task_emb_first, dtype=np.float64)    # (dim,)
    if not task_emb_const:
        # Extremely unlikely for single-task data; fall back to per-query storage.
        print("WARNING: task embedding varies across queries; storing per-query.")
        # Re-read task embeddings per query.
        per_q_task = []
        with open(csv_path, newline="") as f:
            r2 = csv.DictReader(f)
            for i, row in enumerate(r2):
                if i % num_llms == 0:
                    per_q_task.append(_parse_embedding(row["task_description_embedding"]))
        task_emb = np.asarray(per_q_task, dtype=np.float64)    # (num_query, dim)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        query_emb=query_emb,
        task_emb=task_emb,
        effect=np.asarray(effect, dtype=np.float64),
        cost=np.asarray(cost, dtype=np.float64),
        raw_cost=np.asarray(raw_cost, dtype=np.float64),
        split=np.asarray(split, dtype=object).astype("U"),
        llm=np.asarray(llm, dtype=object).astype("U"),
        task_id=np.asarray(task_id, dtype=object).astype("U"),
        metric=np.asarray(metric, dtype=object).astype("U"),
        query_text=np.asarray(query_text, dtype=object).astype("U"),
        ground_truth=np.asarray(ground_truth, dtype=object).astype("U"),
        num_llms=np.int64(num_llms),
        num_query=np.int64(num_query),
    )
    mb_in = os.path.getsize(csv_path) / 1e6
    mb_out = os.path.getsize(out_path) / 1e6
    print(f"{csv_path}\n  -> {out_path}")
    print(f"  rows={num_rows} queries={num_query} llms={num_llms} | "
          f"{mb_in:.0f}MB CSV -> {mb_out:.1f}MB npz ({mb_in / mb_out:.0f}x smaller)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="input selector_data.csv")
    ap.add_argument("--out", required=True, help="output selector_data.npz")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--num-llms", type=int, help="number of workflows per query")
    g.add_argument("--descriptions", help="workflow_descriptions.json (infers num-llms)")
    args = ap.parse_args()

    if args.num_llms:
        num_llms = args.num_llms
    else:
        with open(args.descriptions) as f:
            num_llms = len(json.load(f))
    convert(args.csv, args.out, num_llms)


if __name__ == "__main__":
    main()
