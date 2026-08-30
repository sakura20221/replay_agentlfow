"""Feed the G-Designer family from the shared splits and grade with one scorer.

Mirrors the shape `gsm_data_process` produces -- {"task", "step", "answer"} --
so the authors' training loop needs no changes, and carries the original row
along under "row" because grading MBPP needs its test harness and DROP its
answer spans, neither of which survives being flattened into a string.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
for _ in range(12):
    _ROOT = _ROOT.parent
    if (_ROOT / "shared" / "bench.py").exists():
        break
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

import bench as shared_bench  # noqa: E402

# G-Designer's Graph(domain=...) selects a prompt set. It ships gsm8k, humaneval
# and mmlu; each shared dataset is mapped to whichever of those fits its task
# type, and the mapping is declared rather than silently inherited.
DOMAIN_FOR = {
    "math": "math",
    "amc": "amc",
    "mbpp": "mbpp",
    "drop": "drop",
    "mmlu_pro": "mmlu_pro",
}


def shared_data_process(dataset: list[dict], name: str) -> list[dict]:
    """Convert shared-split records into the loop's expected item shape."""
    items = []
    for row in dataset:
        items.append(
            {
                "task": shared_bench.question_text(name, row),
                "step": "",
                "answer": shared_bench.gold(name, row),
                "row": row,
            }
        )
    return items


def shared_score(name: str, record: dict, prediction: str) -> tuple[float, str]:
    """Grade one prediction through the shared scorer.

    Returns (score, extracted) where score is in [0, 1]; DROP yields fractional
    F1, which flows into the authors' `utility = is_solved` term as partial
    credit instead of being forced to 0/1.
    """
    row = record.get("row") or record
    return shared_bench.score(name, row, prediction or "")


def shared_max_tokens(name: str) -> int:
    return shared_bench.MAX_TOKENS[name]
