#!/usr/bin/env python3
"""Regression test for FlowBank aggregation with duplicate question text."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowbank_pipeline import selected_rounds  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "shims" / "diverseflow" / "aggregate_round_scores.py"


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run(*workflows: tuple[str, Path], out: Path) -> subprocess.CompletedProcess:
    command = [sys.executable, str(SCRIPT)]
    for label, path in workflows:
        command.extend(("--workflow", label, str(path)))
    command.extend(("--out", str(out)))
    return subprocess.run(command, text=True, capture_output=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="flowbank-uid-") as tmp:
        base = Path(tmp)
        duplicate_text = "the same visible question"
        rows = [
            {"uid": "item/1", "question": duplicate_text, "score": 1, "cost": 0},
            {"uid": "item/2", "question": duplicate_text, "score": 0, "cost": 0},
        ]
        first, second = base / "first.jsonl", base / "second.jsonl"
        write_rows(first, rows)
        write_rows(second, list(reversed(rows)))

        complete = run(("Flow_1", first), ("Flow_2", second), out=base / "complete")
        if complete.returncode != 0:
            raise SystemExit(complete.stderr or complete.stdout)
        item_ids = json.loads((base / "complete" / "item_ids.json").read_text())
        if item_ids != ["item/1", "item/2"]:
            raise SystemExit(f"duplicate question text was merged: {item_ids}")

        write_rows(second, rows[:1])
        incomplete = run(("Flow_1", first), ("Flow_2", second), out=base / "incomplete")
        if incomplete.returncode == 0 or "do not cover the same item IDs" not in (
                incomplete.stderr + incomplete.stdout):
            raise SystemExit("missing workflow item was not rejected")

        cura = base / "cura"
        cura.mkdir()
        with (cura / "k_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("k", "best_combo", "oracle_gain_per_query",
                            "oracle_per_query"),
            )
            writer.writeheader()
            writer.writerow({"k": 2, "best_combo": "Flow_1; Flow_3",
                             "oracle_gain_per_query": 0.1,
                             "oracle_per_query": 0.8})
        if selected_rounds(cura, [1, 2, 3], 2) != [1, 3]:
            raise SystemExit("valid CuraFlow portfolio was not preserved")
        try:
            selected_rounds(base / "missing", [1, 2, 3], 2)
        except SystemExit:
            pass
        else:
            raise SystemExit("missing CuraFlow output silently chose a fallback")

    print("FlowBank UID aggregation and portfolio selection OK")


if __name__ == "__main__":
    main()
