"""Benchmark classes that route FlowBank's DiverseFlow stage through the shared scorer.

DiverseFlow is an AFlow fork, so the BaseBenchmark interface is the same and this
file is close to shims/aflow/shared_benchmarks.py. One difference matters a lot
and is the reason these classes exist rather than just data files:

DiverseFlow's own `evaluate_problem` calls `llm_extract_answer(...)` before
scoring, and its MMLU-Pro benchmark additionally reports a `judge_score` from
`llm_judge_answer(...)`. That is an LLM-assisted extraction and an LLM-as-judge
metric. No other method in this comparison gets either, so leaving them in place
would grade FlowBank on a more permissive scale than the other six -- and it also
spends an extra LLM call per problem, which distorts the cost accounting.

These classes therefore bypass both paths and delegate to shared/bench.py, the
single scoring authority for all seven methods. The consequence is that FlowBank's
numbers here are not directly comparable to the ones its own repo reports; that is
intended, and it belongs in the results discussion.
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
    """Generic DiverseFlow benchmark bound to one shared dataset.

    __init__ is inherited: DiverseFlow's BaseBenchmark already accepts the
    llm_config argument the evaluator passes, and it is simply never used here
    because no LLM takes part in scoring.
    """

    SHARED_DATASET: str = ""
    QUESTION_TYPE: str = "math"

    async def _generate_output(self, graph: Callable, input_text: str, entry_point: str | None = None):
        # No tenacity retry wrapper here, unlike the author classes: a retry on a
        # workflow that raises deterministically only multiplies the token spend,
        # and the failure is already recorded as a score of 0 below.
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
        """Interface compatibility only; real grading needs the whole row.

        MBPP needs its test harness and DROP its answer spans, neither of which
        fits this signature, so evaluate_problem calls the shared scorer directly.
        """
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
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill a round
            return input_text, str(exc), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        # Matches the author classes' column set (see benchmarks/math.py) so
        # save_results and the per-problem cost computation keep working.
        return ["question", "prediction", "expected_output", "score", "cost"]


def _make(class_name: str, dataset: str, question_type: str) -> type:
    return type(class_name, (SharedBenchmark,),
                {"SHARED_DATASET": dataset, "QUESTION_TYPE": question_type})


SharedMathBenchmark = _make("SharedMathBenchmark", "math", "math")
SharedAMCBenchmark = _make("SharedAMCBenchmark", "amc", "math")
SharedMBPPBenchmark = _make("SharedMBPPBenchmark", "mbpp", "code")
SharedDROPBenchmark = _make("SharedDROPBenchmark", "drop", "math")
SharedMMLUProBenchmark = _make("SharedMMLUProBenchmark", "mmlu_pro", "math")

# Keys must match DATASETS in shims/diverseflow/install.py: the evaluator turns a
# key into f"datasets/{key.lower()}_{test|validate}.jsonl" at the FlowBank repo
# root, so SHARED_MATH reads shared_math_test.jsonl and cannot collide with the
# authors' own math_test.jsonl.
SHARED_DATASET_CONFIGS = {
    "SHARED_MATH": SharedMathBenchmark,
    "SHARED_AMC": SharedAMCBenchmark,
    "SHARED_MBPP": SharedMBPPBenchmark,
    "SHARED_DROP": SharedDROPBenchmark,
    "SHARED_MMLUPRO": SharedMMLUProBenchmark,
}
