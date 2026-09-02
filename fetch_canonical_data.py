#!/usr/bin/env python3
"""Materialise benchmarks whose frozen files come from canonical upstream data.

MATH and AMC intentionally use the AFlow/FlowBank shipped splits for published
comparison and are therefore not rebuilt here. MBPP, DROP and MMLU-Pro use the
upstream datasets named below. Existing manifest entries are preserved.

Every subsample is a fixed-seed, documented operation, and a manifest with row
counts plus a content hash is written next to the data so a reviewer can verify
the exact evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset  # noqa: E402
from shared.data_utils import mbpp_entry_point, write_manifest

OUT = Path(__file__).resolve().parent / "shared" / "data"
SEED = 20260821
DROP_N = 1000
MMLU_PRO_PER_CATEGORY = 80


def write(name: str, rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
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


def build_mbpp() -> tuple[list[dict], dict]:
    """google-research-datasets/mbpp config=full split=test: official ids 11-510."""
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    rows = []
    for row in ds:
        tests = list(row["test_list"])
        entry_point = mbpp_entry_point(row["code"], tests)
        rows.append(
            {
                "uid": f"mbpp/{row['task_id']}",
                "task_id": int(row["task_id"]),
                "prompt": row["text"],
                "code": row["code"],
                "entry_point": entry_point,
                "test_list": tests,
                "test_imports": list(row.get("test_setup_code", "") and [row["test_setup_code"]] or []),
                "test": "def check():\n" + "".join(f"    {t}\n" for t in tests),
            }
        )
    ids = [r["task_id"] for r in rows]
    meta = {
        "source": "google-research-datasets/mbpp [full/test]",
        "sampling": "none (full canonical split)",
        "task_id_range": [min(ids), max(ids)],
        "inside_official_11_510": sum(1 for i in ids if 11 <= i <= 510),
    }
    return rows, meta


def build_drop() -> tuple[list[dict], dict]:
    """ucinlp/drop validation, fixed-seed subsample (the full dev set is ~9.5k)."""
    ds = load_dataset("ucinlp/drop", split="validation")
    indices = list(range(len(ds)))
    random.Random(SEED).shuffle(indices)
    picked = sorted(indices[:DROP_N])
    rows = []
    for i in picked:
        row = ds[i]
        spans = row["answers_spans"]["spans"]
        rows.append(
            {
                "uid": f"drop/{row.get('query_id', i)}",
                "context": f"Passage: {row['passage']}\nQuestion: {row['question']}",
                "ref_text": "|".join(spans),
                "answers": spans,
            }
        )
    meta = {
        "source": "ucinlp/drop [validation]",
        "sampling": f"uniform random {DROP_N} of {len(ds)}, seed={SEED}",
    }
    return rows, meta


def build_mmlu_pro() -> tuple[list[dict], dict]:
    """TIGER-Lab/MMLU-Pro test, stratified to a fixed count per category."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        by_category[row["category"]].append(row)

    rng = random.Random(SEED)
    rows = []
    for category in sorted(by_category):
        pool = by_category[category]
        rng.shuffle(pool)
        for row in pool[:MMLU_PRO_PER_CATEGORY]:
            rows.append(
                {
                    "uid": f"mmlu_pro/{row['question_id']}",
                    "question": row["question"],
                    "options": list(row["options"]),
                    "answer": row["answer"],
                    "answer_index": int(row["answer_index"]),
                    "category": category,
                }
            )
    meta = {
        "source": "TIGER-Lab/MMLU-Pro [test]",
        "sampling": f"stratified {MMLU_PRO_PER_CATEGORY} per category, seed={SEED}",
        "categories": dict(Counter(r["category"] for r in rows)),
    }
    return rows, meta


BUILDERS = {
    "mbpp": build_mbpp,
    "drop": build_drop,
    "mmlu_pro": build_mmlu_pro,
}
EXPECTED_ROWS = {"mbpp": 500, "drop": DROP_N, "mmlu_pro": 14 * MMLU_PRO_PER_CATEGORY}


def main() -> None:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = SEED
    manifest.setdefault("datasets", {})
    for name, builder in BUILDERS.items():
        print(f"[{name}] fetching ...", flush=True)
        rows, meta = builder()
        if len(rows) != EXPECTED_ROWS[name]:
            raise RuntimeError(
                f"{name}: upstream produced {len(rows)} rows; expected {EXPECTED_ROWS[name]}")
        info = write(name, rows)
        manifest["datasets"][name] = {**info, **meta}
        print(f"[{name}] n={info['n']} sha={info['sha256']}", flush=True)

    total = sum(entry["n"] for entry in manifest["datasets"].values())
    manifest["total_items"] = total
    write_manifest(manifest_path, manifest)
    print(f"\ntotal evaluation items = {total}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
