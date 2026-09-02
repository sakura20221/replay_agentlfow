"""Entry point for DiverseFlow (Stage 1 of FlowBank).

Runs a two-phase MCTS-style search to build a diverse candidate workflow pool:
  - Performance-oriented warm-up for the first ``--diversity_start_round`` rounds.
  - Complementarity-oriented expansion for the remaining rounds.

Defaults match the paper's reported configuration: ``--max_rounds 30``,
``--diversity_start_round 7``, ``--enable_diversity true``.
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
    "DROP": ExperimentConfig(
        dataset="DROP",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble", "QANumerical"],
    ),
    "HotpotQA": ExperimentConfig(
        dataset="HotpotQA",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "AMC": ExperimentConfig(
        dataset="AMC",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "HMMT": ExperimentConfig(
        dataset="HMMT",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "LiveCodeBench": ExperimentConfig(
        dataset="LiveCodeBench",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "Codeforces": ExperimentConfig(
        dataset="Codeforces",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "MMLU_Pro": ExperimentConfig(
        dataset="MMLU_Pro",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="DiverseFlow Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset name",
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="workspace",
        help="Optimized result save path",
    )
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=30,
        help="Total optimization rounds (default 30; paper setting).",
    )
    parser.add_argument(
        "--check_convergence",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable early stop on convergence.",
    )
    parser.add_argument("--validation_rounds", type=int, default=1, help="Validation rounds per round")
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Optimizer LLM name (must be defined in config/config.example.yaml).",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Executor LLM name (must be defined in config/config.example.yaml).",
    )
    # Two-phase diversity-aware optimization (paper Section 3.1).
    parser.add_argument(
        "--enable_diversity",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable two-phase diversity-aware optimization (default True).",
    )
    parser.add_argument(
        "--diversity_start_round",
        type=int,
        default=7,
        help="Round at which complementarity-oriented expansion begins. "
             "Rounds [1, diversity_start_round] form the performance-oriented warm-up "
             "(default 7; paper setting N0=7 over a total budget of N=30).",
    )
    return parser.parse_args()



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
    args = parse_args()

    config = EXPERIMENT_CONFIGS[args.dataset]

    models_config = LLMsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' "
            f"section of the configuration file. Please add it or pass --opt_model_name."
        )

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' "
            f"section of the configuration file. Please add it or pass --exec_model_name."
        )

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
        enable_diversity=args.enable_diversity,
        diversity_start_round=args.diversity_start_round,
    )

    optimizer.optimize("Graph")
    # --- shared-layer shim (agent_wf_v2) --- hard exit v1
    import os as _shim_os, sys as _shim_sys
    _shim_sys.stdout.flush(); _shim_sys.stderr.flush()
    _shim_os._exit(0)
