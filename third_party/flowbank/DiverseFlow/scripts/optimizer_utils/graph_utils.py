import json
import os
import re
import sys
import time
import traceback
from typing import List

from scripts.prompts.optimize_prompt import (
    WORKFLOW_GOAL,
    WORKFLOW_CONTEXT,
    WORKFLOW_DIVERSITY_CONTEXT,
    WORKFLOW_RULES,
    WORKFLOW_TEMPLATE,
    CALL_SIGNATURES,
    generate_operator_usage,
    get_dataset_description,
)
from scripts.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading graph for round {round_number}: {e}")
            raise

    def invalidate_graph_cache(self, round_number: int, workflows_path: str, optimized_path: str = "workspace"):
        """Remove cached modules for a round so that updated graph.py/prompt.py can be reimported."""
        dot_path = workflows_path.replace("\\", ".").replace("/", ".")
        prefixes = [
            f"{dot_path}.round_{round_number}",
        ]
        to_remove = [key for key in sys.modules if any(key.startswith(p) for p in prefixes)]
        for key in to_remove:
            del sys.modules[key]
        if to_remove:
            logger.info(f"Invalidated {len(to_remove)} cached modules for round {round_number}: {to_remove}")

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.error(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        type: str,
        log_data: str,
        operators: List[str] = None,
        dataset: str = "",
        diversity_context: str = "",
    ) -> str:
        # Assembly order: goal FIRST → operator usage → context → diversity → rules LAST
        call_signature = CALL_SIGNATURES.get(type, CALL_SIGNATURES["math"])
        dataset_desc = get_dataset_description(dataset)
        goal = WORKFLOW_GOAL.format(type=type, dataset_description=dataset_desc)
        operator_usage = generate_operator_usage(operators) if operators else ""
        context = WORKFLOW_CONTEXT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            log=log_data,
        )
        rules = WORKFLOW_RULES.format(call_signature=call_signature)
        return goal + operator_usage + context + diversity_context + rules

    def format_diversity_context(self, weight_distribution: dict,
                                  high_weight_examples: List[str],
                                  existing_strengths: str) -> str:
        """Format the diversity context section for Phase 2 prompt injection.

        Args:
            weight_distribution: From coverage_utils.get_weight_distribution().
            high_weight_examples: From coverage_utils.sample_high_weight_queries().
            existing_strengths: From coverage_utils.get_existing_workflow_strengths().
        """
        if not weight_distribution or weight_distribution.get("n_total", 0) == 0:
            return ""

        examples_text = ""
        for i, example in enumerate(high_weight_examples, 1):
            examples_text += f"\n{i}. {example}\n"
        if not examples_text:
            examples_text = "(No specific examples available)"

        return WORKFLOW_DIVERSITY_CONTEXT.format(
            n_weight_high=weight_distribution["n_weight_high"],
            n_weight_mid=weight_distribution["n_weight_mid"],
            n_weight_low=weight_distribution["n_weight_low"],
            high_weight_examples=examples_text,
            existing_strengths=existing_strengths,
        )

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.error(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str, optimized_path: str = "workspace"):
        graph = WORKFLOW_TEMPLATE.format(graph=response["graph"], round=round_number, dataset=dataset, optimized_path=optimized_path)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
