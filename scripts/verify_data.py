#!/usr/bin/env python3
"""Verify every frozen split and all three manifest views agree."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "shared" / "data"
sys.path.insert(0, str(ROOT))

from shared.data_utils import mbpp_entry_point, normalized_task_text  # noqa: E402


DATASETS = {
    "math": ("problem", {"uid", "problem", "answer"}),
    "amc": ("problem", {"uid", "problem", "answer"}),
    "mbpp": ("prompt", {"uid", "prompt", "code", "entry_point", "test_list"}),
    "drop": ("context", {"uid", "context", "ref_text", "answers"}),
    "mmlu_pro": ("question", {"uid", "question", "options", "answer", "category"}),
}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["file_integrity"]["files"]
    failures = []
    for name, record in expected.items():
        path = DATA / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = rows == record["rows"] and digest == record["sha256"]
        print(f"[{'ok' if ok else 'FAIL'}] {name}: rows={rows} sha256={digest[:16]}")
        if not ok:
            failures.append(
                f"{name}: expected rows={record['rows']} sha256={record['sha256']}"
            )

    declared = {}
    for section_name in ("datasets", "search_splits"):
        for key, record in manifest.get(section_name, {}).items():
            name = record.get("file")
            if not name:
                failures.append(f"{section_name}.{key}: missing file")
                continue
            if name in declared:
                failures.append(
                    f"{section_name}.{key}: {name} already declared by {declared[name]}"
                )
            declared[name] = f"{section_name}.{key}"
            integrity = expected.get(name)
            if integrity is None:
                failures.append(f"{section_name}.{key}: {name} absent from file_integrity")
                continue
            if record.get("n") != integrity.get("rows"):
                failures.append(
                    f"{section_name}.{key}: n={record.get('n')} but "
                    f"file_integrity rows={integrity.get('rows')}"
                )
            if record.get("sha256") != integrity.get("sha256", "")[:16]:
                failures.append(
                    f"{section_name}.{key}: sha256={record.get('sha256')} but "
                    f"file_integrity starts {integrity.get('sha256', '')[:16]}"
                )

    eval_total = sum(record.get("n", 0) for record in manifest.get("datasets", {}).values())
    if manifest.get("total_items") != eval_total:
        failures.append(
            f"total_items={manifest.get('total_items')} but dataset total={eval_total}"
        )

    for dataset, (text_field, required) in DATASETS.items():
        eval_rows = read_rows(DATA / f"{dataset}.jsonl")
        search_rows = read_rows(DATA / f"{dataset}_search.jsonl")
        combined = search_rows + eval_rows
        missing_fields = [row.get("uid", "<no uid>") for row in combined
                          if not required <= row.keys()]
        if missing_fields:
            failures.append(f"{dataset}: missing required fields in {missing_fields[:5]}")

        uids = [str(row.get("uid", "")) for row in combined]
        if not all(uids) or len(set(uids)) != len(uids):
            failures.append(f"{dataset}: empty or duplicate uid")

        search_text = {normalized_task_text(row[text_field]) for row in search_rows}
        eval_text = {normalized_task_text(row[text_field]) for row in eval_rows}
        overlap = search_text & eval_text
        if overlap:
            failures.append(f"{dataset}: {len(overlap)} search/eval content overlap(s)")

        train_then_eval = read_rows(DATA / f"{dataset}_train_then_eval.jsonl")
        if train_then_eval != combined:
            failures.append(f"{dataset}: train_then_eval is not exact search + evaluation")

        if dataset == "mbpp":
            bad_entries = [row["uid"] for row in combined
                           if row["entry_point"] !=
                           mbpp_entry_point(row["code"], row["test_list"])]
            if bad_entries:
                failures.append(f"mbpp: wrong entry_point in {bad_entries[:5]}")

        if dataset == "mmlu_pro":
            categories = Counter(row["category"] for row in search_rows)
            if len(categories) != 14 or set(categories.values()) != {18}:
                failures.append(f"mmlu_pro_search: expected 14 x 18, got {dict(categories)}")

        print(f"[ok] {dataset}: search={len(search_rows)} eval={len(eval_rows)} "
              f"cross-overlap={len(overlap)} uid-unique={len(set(uids)) == len(uids)}")

    extras = sorted(path.name for path in DATA.glob("*.jsonl") if path.name not in expected)
    if extras:
        failures.append("unlisted JSONL files: " + ", ".join(extras))
    if failures:
        raise SystemExit("data verification failed:\n  " + "\n  ".join(failures))
    print(f"verified {len(expected)} frozen data files")


if __name__ == "__main__":
    main()
