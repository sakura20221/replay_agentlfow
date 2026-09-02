"""Aggregate FlowBank workflow scores by stable benchmark item UID."""

import argparse
import glob
import json
import os
from collections import defaultdict


def _iter_rows(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        import pandas as pd

        for _, row in pd.read_csv(path).iterrows():
            yield row.to_dict()


def load_workflow_scores(path):
    """Return ``uid -> (mean score, mean cost, question text)``."""
    files = (
        sorted(glob.glob(os.path.join(path, "*.csv"))
               + glob.glob(os.path.join(path, "*.jsonl")))
        if os.path.isdir(path) else [path]
    )
    if not files:
        raise FileNotFoundError(f"no .csv/.jsonl eval files found at {path}")

    score_acc = defaultdict(float)
    cost_acc = defaultdict(float)
    count = defaultdict(int)
    query_text = {}
    for file_path in files:
        seen_in_file = set()
        for row in _iter_rows(file_path):
            question = str(row.get("question", row.get("inputs", "")))
            if not question or "score" not in row:
                continue
            key = str(row.get("uid") or question)
            if key in seen_in_file:
                raise ValueError(
                    f"{file_path}: duplicate item key {key!r}; shared runs must record uid"
                )
            seen_in_file.add(key)
            query_text.setdefault(key, question)
            if query_text[key] != question:
                raise ValueError(
                    f"{file_path}: item {key!r} has inconsistent question text"
                )
            score_acc[key] += float(row["score"])
            cost_acc[key] += float(row.get("cost", 0.0) or 0.0)
            count[key] += 1

    return {
        key: (
            score_acc[key] / count[key],
            cost_acc[key] / count[key],
            query_text[key],
        )
        for key in count
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PATH"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_workflow = {
        label: load_workflow_scores(path) for label, path in args.workflow
    }
    common = None
    for scores in per_workflow.values():
        keys = set(scores)
        common = keys if common is None else common & keys
    if not common:
        raise ValueError("no item IDs are shared across all workflows")

    item_ids = sorted(common)
    dropped = {
        label: len(scores) - len(item_ids)
        for label, scores in per_workflow.items()
    }
    if any(dropped.values()):
        raise ValueError(
            f"workflows do not cover the same item IDs; common={len(item_ids)}, "
            f"dropped per workflow={dropped}"
        )

    first = next(iter(per_workflow.values()))
    questions = [first[item_id][2] for item_id in item_ids]
    sources = {
        label: [scores[item_id][0] for item_id in item_ids]
        for label, scores in per_workflow.items()
    }
    costs = {
        label: [scores[item_id][1] for item_id in item_ids]
        for label, scores in per_workflow.items()
    }

    os.makedirs(args.out, exist_ok=True)
    for name, value in (
        ("sources.json", sources),
        ("costs.json", costs),
        ("queries.json", questions),
        ("item_ids.json", item_ids),
    ):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    print(
        f"Wrote {len(per_workflow)} workflows x {len(item_ids)} queries "
        f"-> {args.out}/"
    )


if __name__ == "__main__":
    main()
