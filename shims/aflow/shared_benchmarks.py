"""Benchmark classes that route AFlow's own runs through the shared scorer.

AFlow's `benchmarks/{math,amc,mbpp,drop}.py` are the metric primitives the shared
layer already delegates to, so it looked at first as though AFlow needed only
data and registration. It does not: for MATH, AMC and MBPP the authors'
`evaluate_problem` reaches `calculate_score` on the same path the shared layer
takes, but DROP diverges -- AFlow splits the model's reply on "|" and maximises
F1 over the parts, while the shared layer first extracts the stated answer span.
Those give different numbers, which would leave DROP graded one way for AFlow and
another way for every other method.

So `evaluate_problem` is overridden here for all five datasets. The metric
definitions stay the authors' (the shared layer calls their `calculate_score`
and `check_solution` underneath); only answer extraction and aggregation become
uniform. MMLU-Pro has no AFlow benchmark at all and is implemented here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable, List, Tuple

from benchmarks.benchmark import BaseBenchmark

_ROOT = Path(__file__).resolve()
for _ in range(12):
    _ROOT = _ROOT.parent
    if (_ROOT / "shared" / "bench.py").exists():
        break
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

import bench as shared_bench  # noqa: E402

TIMEOUT_SECONDS = 1500
MISMATCH_THRESHOLD = 0.3


class SharedBenchmark(BaseBenchmark):
    """Generic AFlow benchmark bound to one shared dataset."""

    SHARED_DATASET: str = ""
    QUESTION_TYPE: str = "math"

    async def _generate_output(self, graph: Callable, input_text: str, entry_point: str | None = None):
        if entry_point is None:
            return await asyncio.wait_for(graph(input_text), timeout=TIMEOUT_SECONDS)
        return await asyncio.wait_for(graph(input_text, entry_point), timeout=TIMEOUT_SECONDS)

    async def load_data(self, specific_indices=None):
        """Smoke mode: cap the split. Same hook as the MaAS-family shim; without
        it a one-round smoke still evaluated the full 256-item validation set,
        and the maths/code cells blew the smoke timeout doing real work on items
        the gate does not need (aflow/math was at 99% of a full evaluation when
        it was killed). Off unless SHIM_SMOKE_N is set."""
        import os as _smoke_os

        data = await super().load_data(specific_indices)
        cap = _smoke_os.getenv("SHIM_SMOKE_N")
        return data[: int(cap)] if cap else data

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """Interface compatibility only; grading needs the whole row."""
        row = getattr(self, "_current_row", None) or {}
        return shared_bench.score(self.SHARED_DATASET, row, prediction)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        name = self.SHARED_DATASET
        self._current_row = problem
        input_text = shared_bench.question_text(name, problem)
        expected_output = shared_bench.gold(name, problem)
        entry_point = problem.get("entry_point") if self.QUESTION_TYPE == "code" else None

        try:
            output, cost = await self._generate_output(graph, input_text, entry_point)
            if not output:
                raise ValueError("output is empty")
            uni_score, _extracted = shared_bench.score(name, problem, output)
            if uni_score < MISMATCH_THRESHOLD:
                self.log_mismatch(input_text, expected_output, output, output)
            return input_text, output, expected_output, uni_score, cost
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill a run
            return input_text, str(exc), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]


def _make(class_name: str, dataset: str, question_type: str) -> type:
    return type(class_name, (SharedBenchmark,),
                {"SHARED_DATASET": dataset, "QUESTION_TYPE": question_type})


SharedMathBenchmark = _make("SharedMathBenchmark", "math", "math")
SharedAMCBenchmark = _make("SharedAMCBenchmark", "amc", "math")
SharedMBPPBenchmark = _make("SharedMBPPBenchmark", "mbpp", "code")
SharedDROPBenchmark = _make("SharedDROPBenchmark", "drop", "math")
SharedMMLUProBenchmark = _make("SharedMMLUProBenchmark", "mmlu_pro", "math")

# Keys must match DATASETS in shims/aflow/install.py: the evaluator turns a key
# into f"data/datasets/{key.lower()}_{test|validate}.jsonl".
SHARED_DATASET_CONFIGS = {
    "SHARED_MATH": SharedMathBenchmark,
    "SHARED_AMC": SharedAMCBenchmark,
    "SHARED_MBPP": SharedMBPPBenchmark,
    "SHARED_DROP": SharedDROPBenchmark,
    "SHARED_MMLUPRO": SharedMMLUProBenchmark,
}
