# -*- coding: utf-8 -*-
# @Desc    : Evaluation for different datasets

import os
from typing import Dict, Literal, Tuple

from benchmarks.amc import AMCBenchmark
from benchmarks.benchmark import BaseBenchmark
from benchmarks.codeforces import CodeforcesBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hmmt import HMMTBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.livecodebench import LiveCodeBench
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark
from benchmarks.mmlu_pro import MMLUProBenchmark

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "AMC", "HMMT", "HotpotQA", "DROP", "LiveCodeBench", "Codeforces", "MMLU_Pro", "SHARED_MATH", "SHARED_AMC", "SHARED_MBPP", "SHARED_DROP", "SHARED_MMLUPRO"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "AMC": AMCBenchmark,
            "HMMT": HMMTBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
            "LiveCodeBench": LiveCodeBench,
            "Codeforces": CodeforcesBenchmark,
            "MMLU_Pro": MMLUProBenchmark,
        }
        # --- shared-layer shim (agent_wf_v2) ---
        from benchmarks.shared_benchmarks import SHARED_DATASET_CONFIGS
        self.dataset_configs.update(SHARED_DATASET_CONFIGS)

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        llm_config = params.get("llm_config", None)
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path, llm_config=llm_config)

        # Use params to configure the graph and benchmark
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None  # For test data, generally use None to test all
        else:
            va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(configured_graph, va_list)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        suffix = "_test.jsonl" if test else "_validate.jsonl"
        fname = f"{dataset.lower()}{suffix}"
        # Resolve the JSONL splits robustly: the repo ships them at the top-level
        # datasets/ dir (this file is DiverseFlow/scripts/evaluator.py, so the
        # repo root is three levels up). Also accept a few standalone layouts.
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(here))            # .../FlowBank
        candidates = [
            os.path.join(repo_root, "datasets", fname),               # shared top-level
            os.path.join(os.path.dirname(here), "datasets", fname),    # DiverseFlow/datasets
            os.path.join(os.path.dirname(here), "data", "datasets", fname),
            os.path.join("data", "datasets", fname),                   # legacy cwd-relative
            os.path.join("datasets", fname),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # Fall back to the top-level path for a clear error message downstream.
        return candidates[0]
