"""Feed MasRouter from the shared splits and grade with the one shared scorer.

MasRouter classifies each query into one of three task types and trains that
classifier with a cross-entropy loss, so every dataset needs a label from its
taxonomy (Math / Commonsense / Code) rather than the hardcoded 0 the MATH runner
uses. Records keep the original row under "row" because MBPP needs its test
harness and DROP its answer spans to be graded at all.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve()
for _ in range(12):
    _ROOT = _ROOT.parent
    if (_ROOT / "shared" / "bench.py").exists():
        break
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

import bench as shared_bench  # noqa: E402

# Index into MAR/Prompts/tasks_profile.py: 0 Math, 1 Commonsense, 2 Code.
TASK_LABEL = {
    "math": 0,
    "amc": 0,
    "mbpp": 2,
    "drop": 1,
    "mmlu_pro": 1,
}

_SPLIT_FILE = {"train": "{name}_search.jsonl", "test": "{name}.jsonl"}


def load_shared_dataset(name: str, split: str = "train") -> list[dict]:
    """Load one shared split in the shape the runner's batch loop expects.

    Returns records with "problem" (the query the router sees) and "row" (the
    full record, needed for grading).
    """
    if split not in _SPLIT_FILE:
        raise ValueError(f"split must be train or test, got {split!r}")

    data_dir = Path(__file__).resolve().parent / "shared"
    path = data_dir / _SPLIT_FILE[split].format(name=name)
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append({"problem": shared_bench.question_text(name, row), "row": row})
    cap = os.getenv("SHIM_SMOKE_N")
    if cap:
        # Smoke gate: validate the pipeline on a handful of items. See
        # shims/maas_family/shared_shim.py for the reasoning.
        records = records[: int(cap)]
    return records


_ITEM_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def shared_score(name: str, row: dict, prediction: str) -> float:
    """Score in [0, 1]. DROP returns F1, so this is not always 0 or 1.

    Every scored item is also appended to logs/scored_items_<dataset>.jsonl.
    Upstream MasRouter has no per-item record: its headline number can otherwise
    only be read from the running mean produced by the scorer live during the
    run. This dump carries the full row and raw prediction so collect.py can
    re-grade MasRouter exactly like the other methods.
    Train and test items share the file and are told apart by uid namespace
    ("<dataset>/..." is evaluation, "<dataset>_search/..." is training).
    """
    inner = row.get("row", row)
    value, _extracted = shared_bench.score(name, inner, prediction or "")
    try:
        _ITEM_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (_ITEM_LOG_DIR / f"scored_items_{name}.jsonl").open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"ts": time.time(), "uid": inner.get("uid"), "row": inner,
                 "prediction": str(prediction or ""), "score": float(value)},
                ensure_ascii=False) + "\n")
    except OSError:
        pass
    return float(value)


def shared_task_labels(name: str, batch: list[dict]) -> list[int]:
    return [TASK_LABEL[name] for _ in batch]
