#!/usr/bin/env python3
"""Regression tests for collector failures that must never become scores."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))

import bench  # noqa: E402
import collect  # noqa: E402


def expect_raises(expected: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - assert the externally visible failure
        if expected not in str(exc):
            raise SystemExit(f"wrong collector failure: {type(exc).__name__}: {exc}")
    else:
        raise SystemExit(f"collector accepted invalid result; expected {expected!r}")


def main() -> None:
    expect_raises(
        "lacks Response or Answer",
        lambda: collect.regrade("math", {"Response": "Answer: 1"}),
    )

    with patch.object(bench, "score", side_effect=RuntimeError("scorer exploded")):
        expect_raises(
            "scorer exploded",
            lambda: collect.regrade(
                "math", {"Response": "Answer: 1", "Answer": "1"}
            ),
        )

    with tempfile.TemporaryDirectory(prefix="collector-strictness-") as tmp:
        base = Path(tmp)
        job = base / "repeat1"
        job.mkdir()
        (job / "status").write_text("ok\n", encoding="utf-8")
        result = base / "result.json"
        result.write_text(
            json.dumps([
                {
                    "uid": "math/one",
                    "Response": "Answer: 1",
                    "Answer": "1",
                }
            ]),
            encoding="utf-8",
        )
        with (
            patch.object(collect, "_find_result_file", return_value=result),
            patch.object(collect, "eval_uids", return_value={"math/one", "math/two"}),
            patch.object(collect, "regrade", return_value=1.0),
            patch.object(collect, "RUN_TAG", "formal-regression"),
        ):
            expect_raises(
                "finished result covers 1/2 evaluation items",
                lambda: collect.from_item_records(job, "gdesigner", "math"),
            )

    print("collector strictness OK: missing inputs, scorer failures and partial finals rejected")


if __name__ == "__main__":
    main()
