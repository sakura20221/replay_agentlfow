"""Evaluate selected workflow rounds on the TEST split (Stage 1 -> Stage 3 bridge).

After CuraFlow picks a portfolio, run those workflow rounds on the held-out test
data so the QueryMatching selector has each workflow's per-test-query scores
(DiverseFlow's optimization only produces train/validation scores). The resulting
test scores become the ``test`` rows of the selector dataset
(see ``data_processing/build_selector_data.py``); combined with the train scores,
the selector is trained on train and its test-set metric is reported each epoch.

Results are saved alongside the original workflow files:
  - Summary:    {workspace}/{dataset}/workflows/results_test.json
  - Per-query:  {workspace}/{dataset}/workflows/round_{N}/test_results/{score}_{timestamp}.jsonl

With --n_runs > 1, each run is saved separately:
  - Summary:    {workspace}/{dataset}/workflows/results_test_run_{i}.json
  - Per-query:  {workspace}/{dataset}/workflows/round_{N}/test_results/run_{i}/{score}_{timestamp}.jsonl

Usage:
    python run_test.py --dataset MATH --rounds 8 11 17       # the CuraFlow-selected rounds
    python run_test.py --dataset MBPP --rounds all
    python run_test.py --dataset AMC --rounds 1 2 3 --n_runs 3 --split validate
"""
import argparse
from typing import Dict, List

from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig


class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "HumanEval": ExperimentConfig("HumanEval", "code", ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
    "MBPP": ExperimentConfig("MBPP", "code", ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
    "GSM8K": ExperimentConfig("GSM8K", "math", ["Custom", "ScEnsemble", "Programmer"]),
    "MATH": ExperimentConfig("MATH", "math", ["Custom", "ScEnsemble", "Programmer"]),
    "AMC": ExperimentConfig("AMC", "math", ["Custom", "ScEnsemble", "Programmer"]),
    "HMMT": ExperimentConfig("HMMT", "math", ["Custom", "ScEnsemble", "Programmer"]),
    "DROP": ExperimentConfig("DROP", "qa", ["Custom", "AnswerGenerate", "ScEnsemble", "QANumerical"]),
    "HotpotQA": ExperimentConfig("HotpotQA", "qa", ["Custom", "AnswerGenerate", "ScEnsemble"]),
    "LiveCodeBench": ExperimentConfig("LiveCodeBench", "code", ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
    "Codeforces": ExperimentConfig("Codeforces", "code", ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
    "MMLU_Pro": ExperimentConfig("MMLU_Pro", "qa", ["Custom", "AnswerGenerate", "ScEnsemble"]),
}


# --- shared-layer shim (agent_wf_v2) ---
EXPERIMENT_CONFIGS["SHARED_MATH"] = ExperimentConfig(
    dataset="SHARED_MATH",
    question_type="math",
    operators=['Custom', 'ScEnsemble', 'Programmer'],
)
EXPERIMENT_CONFIGS["SHARED_AMC"] = ExperimentConfig(
    dataset="SHARED_AMC",
    question_type="math",
    operators=['Custom', 'ScEnsemble', 'Programmer'],
)
EXPERIMENT_CONFIGS["SHARED_MBPP"] = ExperimentConfig(
    dataset="SHARED_MBPP",
    question_type="code",
    operators=['Custom', 'CustomCodeGenerate', 'ScEnsemble', 'Test'],
)
EXPERIMENT_CONFIGS["SHARED_DROP"] = ExperimentConfig(
    dataset="SHARED_DROP",
    question_type="qa",
    operators=['Custom', 'AnswerGenerate', 'ScEnsemble', 'QANumerical'],
)
EXPERIMENT_CONFIGS["SHARED_MMLUPRO"] = ExperimentConfig(
    dataset="SHARED_MMLUPRO",
    question_type="qa",
    operators=['Custom', 'AnswerGenerate', 'ScEnsemble'],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--rounds", nargs="+", required=True,
                        help="Round numbers to evaluate (the CuraFlow-selected workflows), or 'all'.")
    parser.add_argument("--exec_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--optimized_path", type=str, default="workspace")
    parser.add_argument("--workflows_dir", type=str, default="workflows",
                        help="Workflow directory name under {optimized_path}/{dataset}/.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validate"],
                        help="Which data split to evaluate on (default: test).")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of times to run each workflow (default: 1). "
                             "Each run is saved to a separate file for later averaging.")
    args = parser.parse_args()

    if args.rounds == ["all"]:
        test_rounds = list(range(1, 22))
    else:
        test_rounds = [int(r) for r in args.rounds]

    print(f"Testing {args.dataset} rounds: {test_rounds}")

    config = EXPERIMENT_CONFIGS[args.dataset]
    models_config = LLMsConfig.default()
    exec_llm_config = models_config.get(args.exec_model_name)

    # opt_llm_config is unused in test mode, but Optimizer requires it.
    opt_llm_config = exec_llm_config

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=config.operators,
        optimized_path=args.optimized_path,
        workflows_dir=args.workflows_dir,
        sample=4,
    )

    is_test = (args.split == "test")
    optimizer.optimize("Test", test_rounds=test_rounds, is_test=is_test, n_runs=args.n_runs)
