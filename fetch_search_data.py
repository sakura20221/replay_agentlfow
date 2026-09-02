#!/usr/bin/env python3
"""Build the search/training splits that workflow optimisers need.

The evaluation files stay untouched. MATH and AMC use the AFlow/FlowBank
validation files frozen in this repository, so this script rebuilds only MBPP,
DROP and MMLU-Pro. Content overlap with held-out data is excluded before
sampling, not merely checked by row ID.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset  # noqa: E402
from shared.data_utils import mbpp_entry_point, normalized_task_text, write_manifest

OUT = Path(__file__).resolve().parent / "shared" / "data"
SEED = 20260821
SEARCH_N = 256

def write(name: str, rows: list[dict]) -> dict:
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
    return {
        "file": path.name,
        "n": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }


def build_mbpp_search() -> tuple[list[dict], dict]:
    with (OUT / "mbpp.jsonl").open(encoding="utf-8") as handle:
        eval_text = {
            normalized_task_text(json.loads(line)["prompt"])
            for line in handle if line.strip()
        }
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    rows = []
    for row in ds:
        tests = list(row["test_list"])
        entry_point = mbpp_entry_point(row["code"], tests)
        rows.append(
            {
                "uid": f"mbpp_search/{row['task_id']}",
                "task_id": int(row["task_id"]),
                "prompt": row["text"],
                "code": row["code"],
                "entry_point": entry_point,
                "test_list": tests,
                "test_imports": list(row.get("test_setup_code", "") and
                                     [row["test_setup_code"]] or []),
                "test": "def check():\n" + "".join(f"    {t}\n" for t in tests),
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = [r for r in rows if normalized_task_text(r["prompt"]) not in eval_text]
    rows = rows[:SEARCH_N]
    return rows, {
        "source": "google-research-datasets/mbpp [full/train]",
        "sampling": f"uniform random {min(SEARCH_N, len(rows))}, seed={SEED}; "
                    "held-out content excluded before selection",
        "content_overlap_with_eval": 0,
    }


def build_drop_search() -> tuple[list[dict], dict]:
    ds = load_dataset("ucinlp/drop", split="train")
    indices = list(range(len(ds)))
    random.Random(SEED).shuffle(indices)
    rows = []
    for i in indices[: SEARCH_N * 2]:
        row = ds[i]
        spans = row["answers_spans"]["spans"]
        if not spans:
            continue
        rows.append(
            {
                "uid": f"drop_search/{row.get('query_id', i)}",
                "context": f"Passage: {row['passage']}\nQuestion: {row['question']}",
                "ref_text": "|".join(spans),
                "answers": spans,
            }
        )
        if len(rows) >= SEARCH_N:
            break
    return rows, {
        "source": "ucinlp/drop [train]",
        "sampling": f"uniform random {SEARCH_N} of {len(ds)}, seed={SEED}",
    }


def build_mmlu_pro_search() -> tuple[list[dict], dict]:
    """Sample from MMLU-Pro test rows that the evaluation split did not take.

    MMLU-Pro's validation split holds only 70 items, too few to search on, and
    the test split has ~12k rows against an evaluation subset of 1120, so a
    disjoint sample of the remainder keeps the distribution while guaranteeing no
    overlap with anything that gets scored.
    """
    # Iterate the file rather than splitlines(): str.splitlines() also breaks on
    # \x85, U+2028 and U+2029, which appear inside MMLU-Pro question text and
    # would cut a JSON record in half.
    with (OUT / "mmlu_pro.jsonl").open(encoding="utf-8") as handle:
        eval_rows = [json.loads(line) for line in handle if line.strip()]
    eval_uids = {row["uid"] for row in eval_rows}
    eval_text = {normalized_task_text(row["question"]) for row in eval_rows}
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        uid = f"mmlu_pro/{row['question_id']}"
        if uid in eval_uids or normalized_task_text(row["question"]) in eval_text:
            continue
        by_category[row["category"]].append(row)

    rng = random.Random(SEED)
    per_category = max(1, SEARCH_N // max(len(by_category), 1))
    rows = []
    for category in sorted(by_category):
        pool = by_category[category]
        rng.shuffle(pool)
        for row in pool[:per_category]:
            rows.append(
                {
                    "uid": f"mmlu_pro_search/{row['question_id']}",
                    "question": row["question"],
                    "options": list(row["options"]),
                    "answer": row["answer"],
                    "answer_index": int(row["answer_index"]),
                    "category": category,
                }
            )
    rng.shuffle(rows)
    overlap = {r["uid"].split("/", 1)[1] for r in rows} & {u.split("/", 1)[1] for u in eval_uids}
    return rows, {
        "source": "TIGER-Lab/MMLU-Pro [test, disjoint from evaluation subset]",
        "sampling": f"stratified {per_category} per category, seed={SEED}",
        "overlap_with_eval": len(overlap),
        "content_overlap_with_eval": 0,
        "order": "stratified per category, then globally shuffled with the same seed",
    }


BUILDERS = {
    "mbpp_search": build_mbpp_search,
    "drop_search": build_drop_search,
    "mmlu_pro_search": build_mmlu_pro_search,
}
EXPECTED_ROWS = {"mbpp_search": SEARCH_N, "drop_search": SEARCH_N,
                 "mmlu_pro_search": 14 * (SEARCH_N // 14)}


def main() -> None:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("search_splits", {})

    for name, builder in BUILDERS.items():
        print(f"[{name}] building ...", flush=True)
        rows, meta = builder()
        if len(rows) != EXPECTED_ROWS[name]:
            raise RuntimeError(
                f"{name}: upstream produced {len(rows)} rows; expected {EXPECTED_ROWS[name]}")
        info = write(name, rows)
        manifest["search_splits"][name] = {**info, **meta}
        print(f"[{name}] n={info['n']} sha={info['sha256']}", flush=True)

    write_manifest(manifest_path, manifest)
    print(f"\nmanifest updated -> {manifest_path}")


if __name__ == "__main__":
    main()
