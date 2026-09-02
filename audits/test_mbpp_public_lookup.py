#!/usr/bin/env python3
"""Verify every shared MBPP item resolves its own public tests."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "shared" / "data"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows() -> list[dict]:
    result = []
    for split in ("mbpp.jsonl", "mbpp_search.jsonl"):
        with (DATA / split).open(encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle if line.strip())
    return result


def expected_tests(row: dict) -> list[str]:
    entry = row["entry_point"]
    return [re.sub(rf"\b{re.escape(entry)}\b", "candidate", test)
            for test in row["test_list"]]


def verify(label: str, repo: Path, module_path: Path, dataset: str) -> int:
    previous = Path.cwd()
    os.chdir(repo)
    try:
        module = load_module(f"_lookup_{label}", module_path)
        lookup = module.extract_test_cases_from_jsonl
        failures = []
        all_rows = rows()
        for row in all_rows:
            problem = (row["prompt"] + "\n\nYour code must pass these tests:\n"
                       + "\n".join(row["test_list"]))
            actual = lookup(row["entry_point"], dataset=dataset, problem=problem)
            if actual != expected_tests(row):
                failures.append(row["uid"])

        counts = Counter(row["entry_point"] for row in all_rows)
        ambiguous = next(entry for entry, count in counts.items() if count > 1)
        if lookup(ambiguous, dataset=dataset) is not None:
            failures.append(f"ambiguous-name:{ambiguous}")
    finally:
        os.chdir(previous)

    print(f"[{'ok' if not failures else 'FAIL'}] {label}: "
          f"{len(all_rows) - len(failures)}/{len(all_rows)} task identities")
    for failure in failures[:10]:
        print(f"  {failure}")
    return len(failures)


def main() -> None:
    checks = (
        ("maas", ROOT / "third_party" / "maas",
         ROOT / "third_party/maas/maas/ext/maas/scripts/utils.py", "SHARED_MBPP"),
        ("daao", ROOT / "third_party" / "daao",
         ROOT / "third_party/daao/daao/ext/maas/scripts/utils.py", "SHARED_MBPP"),
        ("aflow", ROOT / "third_party" / "aflow",
         ROOT / "third_party/aflow/scripts/utils/code.py", "SHARED_MBPP"),
        ("flowbank", ROOT / "third_party" / "flowbank",
         ROOT / "third_party/flowbank/DiverseFlow/scripts/utils/code.py", "SHARED_MBPP"),
    )
    failures = sum(verify(*check) for check in checks)
    if failures:
        raise SystemExit(f"MBPP public-test lookup: {failures} failure(s)")
    print("MBPP public-test lookup OK")


if __name__ == "__main__":
    main()
