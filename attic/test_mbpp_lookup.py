#!/usr/bin/env python3
"""Resolve the MBPP test lookup the way the running job resolves it.

Two earlier fixes passed a static check and failed at runtime:

* completing `mbpp_public_test.jsonl` -- the code path never opened that file;
* setting `dataset="MBPP"` in SHARED_MBPP's own template -- nothing imports that
  copy, because the seeded graph imports `optimized.HumanEval.train.template`.

So this test does not grep. It imports the operator module the graph imports,
calls the lookup with real MBPP function names, and asserts a list comes back.

    cd third_party/maas && PYTHONPATH=. python ../../test_mbpp_lookup.py maas
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

PKG = sys.argv[1] if len(sys.argv) > 1 else "maas"
SHARED = Path(__file__).resolve().parent / "shared" / "data"


def main() -> int:
    # The runner sets this per job; the lookup reads it to pick the benchmark.
    os.environ["SHIM_DATASET"] = "mbpp"

    graph_module = f"{PKG}.ext.maas.scripts.optimized.SHARED_MBPP.train.graph"
    graph = importlib.import_module(graph_module)
    operator = sys.modules[graph.operator.__name__]
    print(f"  the graph imports its operators from:\n    {operator.__file__}")

    utils = importlib.import_module(f"{PKG}.ext.maas.scripts.utils")
    resolved = utils.extract_test_cases_from_jsonl.__module__
    print(f"  lookup function lives in: {resolved}")

    helper = getattr(operator, "_shim_code_dataset", None)
    if helper is None:
        print("  FAIL: the operator module has no _shim_code_dataset helper")
        return 1
    print(f"  _shim_code_dataset() with SHIM_DATASET=mbpp -> {helper()!r}")
    if helper() != "MBPP":
        print("  FAIL: the helper does not select MBPP")
        return 1

    # Real function names from both splits, including one from each end.
    names = []
    for split in ("mbpp.jsonl", "mbpp_search.jsonl"):
        rows = [json.loads(line) for line in
                (SHARED / split).read_text(encoding="utf-8").splitlines() if line.strip()]
        names += [rows[0]["entry_point"], rows[len(rows) // 2]["entry_point"],
                  rows[-1]["entry_point"]]

    failures = 0
    for name in names:
        try:
            cases = utils.extract_test_cases_from_jsonl(name, dataset=helper())
        except Exception as exc:  # noqa: BLE001 - a raise is the bug under test
            print(f"  FAIL {name:<24} raised {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if not cases:
            print(f"  FAIL {name:<24} returned {cases!r} -- the operator would iterate None")
            failures += 1
            continue
        print(f"  ok   {name:<24} {len(cases)} test case(s), first: {str(cases[0])[:60]!r}")

    # And the HumanEval default must be untouched, so real HumanEval runs still work.
    os.environ["SHIM_DATASET"] = "humaneval"
    if helper() != "HumanEval":
        print("  FAIL: the helper no longer returns HumanEval for HumanEval")
        failures += 1
    else:
        print("  ok   HumanEval still selects HumanEval")

    print(f"\n  -> {failures} failure(s) of {len(names) + 1} check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
