from typing import Dict, List

class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators

EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop"],
    ),
}

# --- shared-layer shim (agent_wf_v2) ---
EXPERIMENT_CONFIGS["SHARED_MATH"] = ExperimentConfig(
    dataset="SHARED_MATH",
    question_type="math",
    operators=['Generate', 'GenerateCoT', 'MultiGenerateCoT', 'ScEnsemble', 'Programmer', 'SelfRefine'],
)
EXPERIMENT_CONFIGS["SHARED_AMC"] = ExperimentConfig(
    dataset="SHARED_AMC",
    question_type="math",
    operators=['Generate', 'GenerateCoT', 'MultiGenerateCoT', 'ScEnsemble', 'Programmer', 'SelfRefine'],
)
EXPERIMENT_CONFIGS["SHARED_MBPP"] = ExperimentConfig(
    dataset="SHARED_MBPP",
    question_type="code",
    operators=['Generate', 'GenerateCoT', 'MultiGenerateCoT', 'ScEnsemble', 'Test', 'SelfRefine'],
)
EXPERIMENT_CONFIGS["SHARED_DROP"] = ExperimentConfig(
    dataset="SHARED_DROP",
    question_type="math",
    operators=['Generate', 'GenerateCoT', 'MultiGenerateCoT', 'ScEnsemble', 'Programmer', 'SelfRefine'],
)
EXPERIMENT_CONFIGS["SHARED_MMLUPRO"] = ExperimentConfig(
    dataset="SHARED_MMLUPRO",
    question_type="math",
    operators=['Generate', 'GenerateCoT', 'MultiGenerateCoT', 'ScEnsemble', 'Programmer', 'SelfRefine'],
)
