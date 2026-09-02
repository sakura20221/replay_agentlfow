# -*- coding: utf-8 -*-
# @Desc    : Inference interface for a previously optimized workflow.

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.logs import logger
from scripts.async_llm import LLMsConfig


def load_best_round(dataset: str, optimized_path: str = "workspace") -> int:
    """Return the round index with the highest validation score."""
    data_utils = DataUtils(f"{optimized_path}/{dataset}")

    # Pick the top-scoring round via DataUtils.get_top_rounds.
    top_rounds = data_utils.get_top_rounds(sample=2, mode="Graph")
    if not top_rounds[1]:
        return 1

    return top_rounds[1]["round"]


def load_workflow_class(graph_path: str):
    """Dynamically import the Workflow class from a generated graph.py file."""
    spec = importlib.util.spec_from_file_location("workflow_module", graph_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_module"] = module
    spec.loader.exec_module(module)
    return module.Workflow


async def workflow_inference(
    dataset: DatasetType,
    question: str,
    entry_point: Optional[str] = None,
    round: Optional[int] = None,
    llm_name: str = "gpt-4o-mini",
    optimized_path: str = "workspace",
) -> Tuple[str, float]:
    """Run inference with a previously optimized workflow.

    Args:
        dataset: dataset name.
        question: input query.
        entry_point: function name (required for code datasets).
        round: round index to use; falls back to the best round when None.
        llm_name: executor LLM name as defined in the loaded config file.
        optimized_path: directory where optimization artifacts are stored.

    Returns:
        Tuple of ``(answer, cost)``.
    """
    # Fall back to the best round when none is specified.
    if round is None:
        round = load_best_round(dataset, optimized_path)

    logger.info(f"Using round {round} for inference")

    # Locate the generated workflow file.
    graph_path = Path(optimized_path) / dataset / "workflows" / f"round_{round}" / "graph.py"
    if not graph_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {graph_path}")

    # Dynamically load the Workflow class produced during optimization.
    WorkflowClass = load_workflow_class(str(graph_path))

    # Instantiate the workflow with the configured executor LLM.
    llm_config = LLMsConfig.default().get(llm_name)
    workflow = WorkflowClass(
        name=f"{dataset}_workflow",
        llm_config=llm_config,
        dataset=dataset,
    )

    # Run inference. Code datasets require the additional ``entry_point`` arg.
    if dataset in ["MBPP", "HumanEval"]:
        answer, cost = await workflow(question, entry_point=entry_point)
    else:
        answer, cost = await workflow(question)

    return answer, cost


if __name__ == "__main__":
    asyncio.run(
        workflow_inference(
            dataset="MBPP",
            question="write a function named add_two_numbers to calculate the sum of two numbers",
            entry_point="add_two_numbers",
        )
    )
