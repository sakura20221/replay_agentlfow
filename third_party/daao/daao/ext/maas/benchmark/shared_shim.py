"""Benchmark adapters that let the MaAS family score against the shared layer.

Installed into both MaAS and DAAO, which are structurally the same repo (DAAO is
a fork) apart from one thing: DAAO's `_generate_output` also returns a `vae`
dict, and its `evaluate_problem` writes `vae["is_solved"]` back from the score.
That field is the training signal for DAAO's difficulty VAE, so dropping it would
leave the difficulty estimator learning nothing while still appearing to run.
The arity is therefore detected at runtime instead of being assumed.

Scoring never happens here: it is delegated to shared/bench.py so that all seven
methods in the bake-off are graded by one code path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, List

import torch

from .benchmark import BaseBenchmark

# shared/bench.py lives outside the vendored repo; add the experiment root.
_EXPERIMENT_ROOT = Path(__file__).resolve()
for _ in range(12):
    _EXPERIMENT_ROOT = _EXPERIMENT_ROOT.parent
    if (_EXPERIMENT_ROOT / "shared" / "bench.py").exists():
        break
if str(_EXPERIMENT_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT / "shared"))

import bench as shared_bench  # noqa: E402

logger = logging.getLogger("shared_shim")
# Set SHIM_TRACEBACKS=0 once a run is known-good; during bring-up the stack
# is the only way to tell a real failure from a swallowed one.
SHIM_TRACEBACKS = os.getenv("SHIM_TRACEBACKS", "1") not in {"0", "false", "False"}

TIMEOUT_SECONDS = 1500

# MaAS returns (output, cost, logprob); DAAO returns (output, cost, logprob, vae).
EXPECTS_VAE = (__package__ or "").split(".")[0] == "daao"

VAE_ZERO_DIM = 32


def _zero_vae(device: torch.device) -> dict:
    return {
        "z_difficulty": torch.zeros((1, VAE_ZERO_DIM), device=device),
        "difficulty_scalar": torch.tensor(0.5, device=device),
        "mu": torch.zeros((1, VAE_ZERO_DIM), device=device),
        "logvar": torch.zeros((1, VAE_ZERO_DIM), device=device),
        "is_solved": 0,
    }


class SharedBenchmark(BaseBenchmark):
    async def load_data(self, specific_indices=None):
        """Smoke mode: cap the split so one cell exercises the WHOLE pipeline in
        minutes. The point of a smoke run is to validate plumbing -- prompts,
        proxy, operators, per-item files, collection -- on real code paths; item
        count is irrelevant to that, and 6 items instead of 1,000 is what turns a
        pre-launch gate from hours into minutes. Off unless SHIM_SMOKE_N is set,
        so a full run is byte-identical to before this hook existed.
        """
        import os as _smoke_os

        data = await super().load_data(specific_indices)
        cap = _smoke_os.getenv("SHIM_SMOKE_N")
        return data[: int(cap)] if cap else data

    """Generic benchmark bound to one shared dataset."""

    SHARED_DATASET: str = ""
    QUESTION_TYPE: str = "math"

    async def _generate_output(self, graph: Callable, prompt: str, entry_point: str | None = None):
        if entry_point is None:
            return await asyncio.wait_for(graph(prompt), timeout=TIMEOUT_SECONDS)
        return await asyncio.wait_for(graph(prompt, entry_point, self.log_path), timeout=TIMEOUT_SECONDS)

    def extract_model_answer(self, text: str) -> str:
        _, extracted = shared_bench.score(self.SHARED_DATASET, self._current_row or {}, text)
        return str(extracted)

    def calculate_score(self, expected_output: str, prediction: str):
        """Kept for interface compatibility.

        The real grading needs the whole row (MBPP needs its test harness, DROP
        its answer spans), which this signature cannot carry, so evaluate_problem
        calls the shared scorer directly.
        """
        row = self._current_row or {}
        return shared_bench.score(self.SHARED_DATASET, row, prediction)

    _current_row: dict | None = None

    async def evaluate_problem(self, problem: dict, graph: Callable):
        name = self.SHARED_DATASET
        self._current_row = problem
        input_text = shared_bench.question_text(name, problem)
        expected_output = shared_bench.gold(name, problem)
        entry_point = problem.get("entry_point") if self.QUESTION_TYPE == "code" else None

        try:
            result = await self._generate_output(graph, input_text, entry_point)
            if isinstance(result, (tuple, list)) and len(result) == 4:
                output, cost, logprob, vae = result
            else:
                output, cost, logprob = result
                vae = None
            if not output:
                # An empty final answer is a legitimate negative outcome, not an
                # error: on an 8B executor the operators' XML field extraction
                # often yields nothing, and the workflow returns "". Scoring it 0
                # while keeping the real logprob lets the controller learn that
                # this operator combination produced nothing. Raising instead
                # would discard the gradient and silently shrink the training set
                # -- 11 samples in one smoke run.
                logger.warning(f"[shared_shim] {name}: empty output, scoring 0 with the real gradient")
                empty_out = ""
                if vae is not None:
                    vae["is_solved"] = 0
                    return input_text, empty_out, expected_output, 0.0, cost, logprob, vae
                return input_text, empty_out, expected_output, 0.0, cost, logprob

            uni_score, extracted_output = shared_bench.score(name, problem, output)

            if uni_score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output,
                                  extract_answer_code=f"shared_bench.score({name!r}, row, prediction)")

            if vae is not None:
                vae["is_solved"] = 0 if uni_score == 0 else 1
                return input_text, output, expected_output, uni_score, cost, logprob, vae
            return input_text, output, expected_output, uni_score, cost, logprob

        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill a run
            # Log loudly. The zero logprob returned below carries no gradient, so
            # if every sample lands here the controller silently stops training
            # and the run still completes and reports scores -- observed as
            # "Loss does not require grad and was skipped" with no other clue.
            logger.warning(
                f"[shared_shim] {name} sample failed, returning a zero-gradient "
                f"placeholder: {type(exc).__name__}: {exc}"
            )
            if SHIM_TRACEBACKS:
                logger.warning("[shared_shim] traceback:\n" + traceback.format_exc())
            zero_logprob = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if EXPECTS_VAE:
                return (input_text, str(exc), expected_output, 0.0, 0.0, zero_logprob,
                        _zero_vae(self.device))
            return input_text, str(exc), expected_output, 0.0, 0.0, zero_logprob

    def get_result_columns(self) -> List[str]:
        columns = ["question", "prediction", "expected_output", "score", "cost", "logprob"]
        if EXPECTS_VAE:
            columns.append("vae")
        return columns


def _make(class_name: str, dataset: str, question_type: str) -> type:
    return type(class_name, (SharedBenchmark,),
                {"SHARED_DATASET": dataset, "QUESTION_TYPE": question_type})


# Keys must stay in lockstep with DATASETS in shims/maas_family/install.py: the
# evaluator turns a key into a data path via f"{dataset.lower()}_{split}.jsonl",
# and the SHARED_ prefix is what keeps those paths from colliding with the
# authors' own drop_test.jsonl / mbpp_test.jsonl / mbpp_train.jsonl.
SharedMathBenchmark = _make("SharedMathBenchmark", "math", "math")
SharedAMCBenchmark = _make("SharedAMCBenchmark", "amc", "math")
SharedMBPPBenchmark = _make("SharedMBPPBenchmark", "mbpp", "code")
SharedDROPBenchmark = _make("SharedDROPBenchmark", "drop", "math")
SharedMMLUProBenchmark = _make("SharedMMLUProBenchmark", "mmlu_pro", "math")

SHARED_DATASET_CONFIGS = {
    "SHARED_MATH": SharedMathBenchmark,
    "SHARED_AMC": SharedAMCBenchmark,
    "SHARED_MBPP": SharedMBPPBenchmark,
    "SHARED_DROP": SharedDROPBenchmark,
    "SHARED_MMLUPRO": SharedMMLUProBenchmark,
}
