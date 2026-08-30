#!/usr/bin/env python3
"""Build the search/training splits that workflow optimisers need.

The evaluation splits produced by fetch_canonical_data.py must stay untouched,
so the search sets come from each benchmark's own upstream *train* split rather
than from a slice of the test set. Where no train split exists the fallback is
stated explicitly rather than quietly carved out of the evaluation data.

AMC is the one benchmark with no train split at all -- the canonical set is 83
items and all of them are needed for evaluation -- so methods search on MATH and
transfer to AMC. That is a declared protocol substitution, not a reproduction of
any published AMC setup.
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

OUT = Path(__file__).resolve().parent / "shared" / "data"
SEED = 20260821
SEARCH_N = 256

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def write(name: str, rows: list[dict]) -> dict:
    path = OUT / f"{name}.jsonl"
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
    return {"file": path.name, "n": len(rows), "sha256": digest.hexdigest()[:16]}


def _final_boxed(solution: str) -> str:
    """Pull the answer out of a MATH-style solution's last \\boxed{...}."""
    marker = "\\boxed"
    index = solution.rfind(marker)
    if index < 0:
        return ""
    tail = solution[index + len(marker):]
    if not tail.startswith("{"):
        return tail.strip().split("$")[0].strip()
    depth = 0
    for position, char in enumerate(tail):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return tail[1:position]
    return ""


def build_math_search() -> tuple[list[dict], dict]:
    pool = []
    for config in MATH_CONFIGS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {config}: {type(exc).__name__}")
            continue
        for row in ds:
            answer = _final_boxed(row.get("solution", ""))
            if answer:
                pool.append(
                    {
                        "problem": row["problem"],
                        "answer": answer,
                        "solution": row["solution"],
                        "level": row.get("level"),
                        "type": row.get("type") or config,
                    }
                )
    rng = random.Random(SEED)
    rng.shuffle(pool)
    picked = pool[:SEARCH_N]
    rows = [{"uid": f"math_search/{i}", **row} for i, row in enumerate(picked)]
    return rows, {
        "source": "EleutherAI/hendrycks_math [train, all 7 subjects]",
        "sampling": f"uniform random {SEARCH_N} of {len(pool)}, seed={SEED}",
        "levels": dict(Counter(str(r["level"]) for r in rows)),
    }


def build_mbpp_search() -> tuple[list[dict], dict]:
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    rows = []
    for row in ds:
        tests = list(row["test_list"])
        entry_point = row["code"].split("def ", 1)[1].split("(", 1)[0].strip() if "def " in row["code"] else ""
        rows.append(
            {
                "uid": f"mbpp_search/{row['task_id']}",
                "task_id": int(row["task_id"]),
                "prompt": row["text"],
                "code": row["code"],
                "entry_point": entry_point,
                "test_list": tests,
                "test": "def check():\n" + "".join(f"    {t}\n" for t in tests),
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:SEARCH_N]
    return rows, {
        "source": "google-research-datasets/mbpp [full/train]",
        "sampling": f"uniform random {min(SEARCH_N, len(rows))}, seed={SEED}",
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
        eval_uids = {json.loads(line)["uid"] for line in handle if line.strip()}
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        uid = f"mmlu_pro/{row['question_id']}"
        if uid in eval_uids:
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
    overlap = {r["uid"].split("/", 1)[1] for r in rows} & {u.split("/", 1)[1] for u in eval_uids}
    return rows, {
        "source": "TIGER-Lab/MMLU-Pro [test, disjoint from evaluation subset]",
        "sampling": f"stratified {per_category} per category, seed={SEED}",
        "overlap_with_eval": len(overlap),
    }


# math_search was re-based on the official L5 validate split on 2026-08-24 and
# amc_search on FlowBank's shipped validate; their builders are parked so a
# re-run cannot clobber the adopted files.
BUILDERS = {
    "mbpp_search": build_mbpp_search,
    "drop_search": build_drop_search,
    "mmlu_pro_search": build_mmlu_pro_search,
}


def main() -> None:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("search_splits", {})

    for name, builder in BUILDERS.items():
        print(f"[{name}] building ...", flush=True)
        rows, meta = builder()
        info = write(name, rows)
        manifest["search_splits"][name] = {**info, **meta}
        print(f"[{name}] n={info['n']} sha={info['sha256']}", flush=True)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest updated -> {manifest_path}")


if __name__ == "__main__":
    main()
