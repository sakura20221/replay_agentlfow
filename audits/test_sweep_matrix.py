#!/usr/bin/env python3
"""Static regression for the complete method x dataset execution matrix."""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flowbank_pipeline  # noqa: E402
import sweep  # noqa: E402


def main() -> None:
    failures = []
    expected_methods = {
        "maas", "daao", "gdesigner", "card", "gdesigner_authordefault",
        "card_authordefault", "masrouter", "aflow", "flowbank",
    }
    if set(sweep.METHODS) != expected_methods:
        failures.append(f"method set is {sorted(sweep.METHODS)}")
    if sweep.EXCLUDED:
        failures.append(f"unexpected excluded cells: {sweep.EXCLUDED}")

    for dataset in sweep.DATASETS:
        search_rows = sum(1 for line in
                          (ROOT / "shared/data" / f"{dataset}_search.jsonl").open()
                          if line.strip())
        eval_rows = sum(1 for line in
                        (ROOT / "shared/data" / f"{dataset}.jsonl").open()
                        if line.strip())
        if sweep.TRAIN_ITEMS[dataset] != search_rows:
            failures.append(f"{dataset}: TRAIN_ITEMS != search rows")
        expected_batches = math.ceil(search_rows / 4)
        if sweep.TRAIN_BATCHES[dataset] != expected_batches:
            failures.append(f"{dataset}: TRAIN_BATCHES != ceil(search/4)")
        key = sweep.SHARED_KEY[dataset]
        if flowbank_pipeline.expected_items(key, "train") != search_rows:
            failures.append(f"{dataset}: FlowBank train count lookup failed")
        if flowbank_pipeline.expected_items(key, "test") != eval_rows:
            failures.append(f"{dataset}: FlowBank test count lookup failed")

        for method in expected_methods:
            command = sweep.build(method, dataset, "search")
            if not command:
                failures.append(f"{method}/{dataset}: no search command")
            if method in {"gdesigner", "card"} and (
                    f"--train_items {search_rows}" not in command
                    or f"--search_items {search_rows}" not in command):
                failures.append(f"{method}/{dataset}: wrong held-out boundary")

    for method in ("maas", "daao", "aflow"):
        for dataset in sweep.DATASETS:
            if not sweep.build(method, dataset, "test"):
                failures.append(f"{method}/{dataset}: no held-out test command")

    if failures:
        raise SystemExit("sweep matrix regression failed:\n  " + "\n  ".join(failures))
    print("sweep matrix OK: 9 method rows x 5 datasets; all boundaries resolved")


if __name__ == "__main__":
    main()
