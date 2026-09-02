# -*- coding: utf-8 -*-
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import json
import os
import time
from typing import List, Literal, Dict

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.coverage_utils import CoverageUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.optimizer_utils.visualization_utils import VisualizationUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        workflows_dir: str = "workflows",
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
        enable_diversity: bool = False,
        diversity_start_round: int = None,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.optimized_path = optimized_path
        self.root_path = f"{optimized_path}/{self.dataset}"
        self.workflows_dir = workflows_dir
        self.sample = sample
        self.top_scores = []
        self.initial_round = initial_round
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        # Diversity settings
        self.enable_diversity = enable_diversity
        self.diversity_start_round = diversity_start_round if diversity_start_round is not None else max_rounds // 2

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)
        self.coverage_utils = CoverageUtils(self.root_path)
        self.visualization_utils = VisualizationUtils(self.root_path)

    def _ensure_workspace(self):
        """Ensure workspace directory has template and round_1 for the current dataset.

        When optimized_path differs from 'workspace' (e.g., 'workspace_qwen3'),
        we need to copy the template and round_1 from the default workspace
        and fix the import paths in round_1/graph.py.
        """
        target_workflows = f"{self.root_path}/workflows"
        template_dir = os.path.join(target_workflows, "template")
        round1_dir = os.path.join(target_workflows, "round_1")

        # If template and round_1 already exist with content, nothing to do
        if (os.path.isdir(template_dir) and os.listdir(template_dir)
                and os.path.isdir(round1_dir) and os.path.exists(os.path.join(round1_dir, "graph.py"))):
            return

        # Source from default workspace
        default_root = f"workspace/{self.dataset}"
        default_template = f"{default_root}/workflows/template"
        default_round1 = f"{default_root}/workflows/round_1"

        if not os.path.isdir(default_template):
            logger.warning(f"Default template not found at {default_template}")
            return

        import shutil

        # Copy template
        if not os.path.isdir(template_dir) or not os.listdir(template_dir):
            os.makedirs(template_dir, exist_ok=True)
            shutil.copytree(default_template, template_dir, dirs_exist_ok=True)
            logger.info(f"Copied template from {default_template} to {template_dir}")

        # Copy round_1
        if not os.path.exists(os.path.join(round1_dir, "graph.py")):
            os.makedirs(round1_dir, exist_ok=True)
            shutil.copytree(default_round1, round1_dir, dirs_exist_ok=True)
            logger.info(f"Copied round_1 from {default_round1} to {round1_dir}")

            # Fix import paths in graph.py: workspace.{DATASET} -> {optimized_path}.{DATASET}
            graph_py = os.path.join(round1_dir, "graph.py")
            if os.path.exists(graph_py):
                with open(graph_py, "r") as f:
                    content = f.read()
                content = content.replace(f"import workspace.{self.dataset}.",
                                          f"import {self.optimized_path}.{self.dataset}.")
                with open(graph_py, "w") as f:
                    f.write(content)
                logger.info(f"Fixed import paths in {graph_py}")

        # Ensure __init__.py files exist
        for d in [f"{self.optimized_path}", f"{self.optimized_path}/{self.dataset}",
                   target_workflows]:
            os.makedirs(d, exist_ok=True)
            init_py = os.path.join(d, "__init__.py")
            if not os.path.exists(init_py):
                with open(init_py, "w") as f:
                    f.write("")

    def _detect_resume_round(self):
        """Detect the last completed round from results.json for resume support."""
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)
        if not data:
            return None
        completed_rounds = {d["round"] for d in data}
        max_round = max(completed_rounds)
        logger.info(f"Resume: found {len(completed_rounds)} completed rounds (up to round {max_round})")
        return max_round

    def optimize(self, mode: OptimizerType = "Graph", test_rounds=None, is_test=True, n_runs=1):
        # Ensure workspace has template + round_1 (auto-copies from default workspace if needed)
        self._ensure_workspace()

        if mode == "Test":
            for i in range(n_runs):
                run_id = (i + 1) if n_runs > 1 else None
                if run_id is not None:
                    logger.info(f"=== Run {run_id}/{n_runs} ===")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test(rounds=test_rounds, is_test=is_test, run_id=run_id))
            return None

        # Resume from last completed round if results exist
        last_round = self._detect_resume_round()
        if last_round is not None and last_round >= self.round:
            remaining = self.max_rounds - (last_round - self.initial_round)
            if remaining <= 0:
                logger.info(f"All {self.max_rounds} rounds already completed. Nothing to do.")
                return None
            logger.info(f"Resuming from round {last_round + 1} ({remaining} rounds remaining)")
            self.round = last_round
        else:
            remaining = self.max_rounds

        for opt_round in range(remaining):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

        # Generate visualization after all rounds complete
        self._generate_visualizations()

    def _generate_visualizations(self):
        """Generate MCT tree and performance/cost charts."""
        try:
            self.experience_utils.load_experience()
            self.visualization_utils.plot_mct_tree()
            # Compute final weighted scores for visualization
            if self.enable_diversity:
                weighted_scores = self.coverage_utils.recompute_weighted_scores()
                self.data_utils.update_all_weighted_scores(weighted_scores)
            self.visualization_utils.plot_performance_cost()
            logger.info("Visualization generation completed.")
        except Exception as e:
            logger.warning(f"Visualization generation failed (non-critical): {e}")

    def _is_diversity_mode(self) -> bool:
        """Check if current round should use diversity-aware optimization."""
        if not self.enable_diversity:
            return False
        # Count completed rounds (current round number relative to initial)
        completed = self.round - self.initial_round + 1
        return completed >= self.diversity_start_round

    async def _optimize_graph(self):
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True)

        # Determine phase
        diversity_mode = self._is_diversity_mode()
        if diversity_mode:
            logger.info(f"[Phase 2 - Diversity Mode] Round {self.round + 1}")
            self.coverage_utils.build_coverage_matrix()
            coverage_stats = self.coverage_utils.get_coverage_stats()
            weighted_scores = self.coverage_utils.compute_all_weighted_scores()
            weight_dist = self.coverage_utils.get_weight_distribution()
            logger.info(
                f"[Diversity] Coverage: {coverage_stats['n_covered']}/{coverage_stats['n_total']} "
                f"({coverage_stats['coverage_rate']:.1%}), "
                f"weights: {weight_dist['n_weight_high']} high / {weight_dist['n_weight_mid']} mid / {weight_dist['n_weight_low']} low"
            )
        else:
            weighted_scores = None
            if self.enable_diversity:
                logger.info(f"[Phase 1 - Standard Mode] Round {self.round + 1}")

        # --- Parent Selection ---
        retry_count = 0
        max_retries = 5
        max_total_retries = 10

        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            # Phase 2: rank parents by weighted_score; Phase 1: by raw score
            top_rounds = self.data_utils.get_top_rounds(
                self.sample, use_weighted=diversity_mode
            )
            sample = self.data_utils.select_round(top_rounds)

            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            # --- Experience ---
            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(
                processed_experience, sample["round"],
                use_weighted=diversity_mode
            )

            log_data = self.data_utils.load_log(sample["round"])

            # --- Prompt Assembly ---
            # Phase 2: pass weighted_score to prompt; Phase 1: pass raw score
            prompt_score = sample["score"]  # Already weighted_score if use_weighted=True in _load_scores

            diversity_context = ""
            if diversity_mode:
                high_weight_examples = self.coverage_utils.sample_high_weight_queries(
                    dataset=self.dataset, n=5
                )
                existing_strengths = self.coverage_utils.get_existing_workflow_strengths(
                    weighted_scores=weighted_scores, top_k=5
                )
                diversity_context = self.graph_utils.format_diversity_context(
                    weight_dist, high_weight_examples, existing_strengths
                )

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, prompt_score, graph[0], prompt, self.type, log_data,
                operators=self.operators,
                dataset=self.dataset,
                diversity_context=diversity_context,
            )

            # Dedup instruction after retries
            if retry_count >= max_retries:
                past_mods = self.experience_utils.get_past_modifications(processed_experience, sample["round"])
                if past_mods:
                    dedup_instruction = "\n\n**CRITICAL: Your previous modifications were ALL duplicates. You MUST propose a COMPLETELY DIFFERENT modification. Here are the modifications already tried — do NOT repeat any of them:**\n"
                    for i, mod in enumerate(past_mods, 1):
                        dedup_instruction += f"  {i}. {mod}\n"
                    dedup_instruction += "\nThink of a fundamentally different approach: use different operators, change the workflow structure, modify the number of iterations, or try an entirely new strategy."
                    graph_optimize_prompt += dedup_instruction
                    logger.info(f"Added dedup instruction after {retry_count} retries (parent round {sample['round']})")

            if retry_count >= max_total_retries:
                logger.warning(f"Reached max total retries ({max_total_retries}), forcing acceptance")

            # --- LLM Call ---
            try:
                graph_formatter = XmlFormatter.from_model(GraphOptimize)
                response = await self.optimize_llm.call_with_format(
                    graph_optimize_prompt, graph_formatter
                )
                logger.info(f"Graph optimization response received successfully")
            except FormatError as e:
                logger.error(f"Format error in graph optimization: {str(e)}")
                raw_response = await self.optimize_llm(graph_optimize_prompt)
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    logger.error("Failed to extract fields from raw response, retrying...")
                    retry_count += 1
                    continue

            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )
            if check or retry_count >= max_total_retries:
                break
            retry_count += 1
            logger.info(f"Duplicate modification detected (retry {retry_count}/{max_total_retries}), regenerating...")

        # --- Evaluate ---
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset, self.optimized_path)

        try:
            self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)
            logger.info(directory)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Graph load/compile error: {error_msg}. Attempting auto-fix...")
            avg_score = await self._attempt_autofix(response, graph_path, directory, validation_n, data, error_msg)

        # Runtime error auto-fix
        if avg_score == 0.0:
            runtime_error = self._detect_runtime_error(directory)
            if runtime_error:
                logger.warning(f"All queries failed: {runtime_error[:200]}. Attempting auto-fix...")
                avg_score = await self._attempt_autofix(
                    response, graph_path, directory, validation_n, data, runtime_error
                )

        # --- Recompute weighted scores for ALL rounds (both phases) ---
        if self.enable_diversity:
            weighted_scores = self.coverage_utils.recompute_weighted_scores()
            self.data_utils.update_all_weighted_scores(weighted_scores)
            child_ws = weighted_scores.get(self.round + 1)
            parent_ws = weighted_scores.get(sample["round"])
            if child_ws is not None:
                logger.info(
                    f"[Weighted] Round {self.round + 1}: ws={child_ws:.4f}, "
                    f"parent(round {sample['round']}) ws={parent_ws:.4f}, "
                    f"raw_acc={avg_score:.4f}"
                )

        # --- Experience Update ---
        # In Phase 2, sample["score"] is weighted_score (from use_weighted=True).
        # We need the raw parent score from results.json.
        parent_raw = sample["score"]  # Phase 1: this is raw; Phase 2: this is weighted
        if diversity_mode:
            # Look up the actual raw score from results data
            for entry in data:
                if entry.get("round") == sample["round"]:
                    parent_raw = entry.get("score", sample["score"])
                    break

        experience = self.experience_utils.create_experience_data(sample, response["modification"])
        self.experience_utils.update_experience(
            directory, experience, avg_score,
            parent_raw_score=parent_raw,
            weighted_scores=weighted_scores,
            parent_round=sample["round"],
            child_round=self.round + 1,
            use_weighted_for_success=diversity_mode,
        )

        return avg_score

    async def _fix_graph_error(self, original_response, graph_code, prompt_code, error_msg):
        """Ask the optimizer LLM to fix a graph that failed to load/execute."""
        from scripts.prompts.optimize_prompt import WORKFLOW_FIX_PROMPT, generate_operator_usage, CALL_SIGNATURES

        call_signature = CALL_SIGNATURES.get(self.type, CALL_SIGNATURES["math"])
        operator_usage = generate_operator_usage(self.operators) if self.operators else ""

        fix_prompt = WORKFLOW_FIX_PROMPT.format(
            error_msg=error_msg,
            graph_code=graph_code,
            prompt_code=prompt_code,
            operator_usage=operator_usage,
            call_signature=call_signature,
        )

        try:
            graph_formatter = XmlFormatter.from_model(GraphOptimize)
            fixed_response = await self.optimize_llm.call_with_format(fix_prompt, graph_formatter)
            fixed_response["modification"] = (
                f"[Auto-fixed] Original: {original_response['modification']} | Fix: {fixed_response['modification']}"
            )
            logger.info(f"Auto-fix response received: {fixed_response['modification']}")
            return fixed_response
        except Exception as e:
            logger.error(f"Auto-fix LLM call failed: {e}")
            return None

    def _detect_runtime_error(self, directory: str) -> str:
        """Check if all queries in the latest .jsonl failed with the same runtime error.

        Returns the common error message if all predictions are error strings,
        or empty string if results look normal (some queries have real outputs).
        """
        import glob as glob_mod

        jsonl_files = sorted(
            glob_mod.glob(os.path.join(directory, "*.jsonl")),
            key=os.path.getmtime, reverse=True
        )
        if not jsonl_files:
            return ""

        predictions = []
        try:
            with open(jsonl_files[0], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    predictions.append(record.get("prediction", ""))
        except Exception:
            return ""

        if not predictions:
            return ""

        # Heuristic: known runtime error patterns (both class names and message text)
        error_patterns = [
            "is not defined", "has no attribute", "Timeout",
            "KeyError", "TypeError", "ValueError", "IndexError",
            "AttributeError", "NameError", "ImportError", "ModuleNotFoundError",
            "SyntaxError", "IndentationError",
            "out of range", "Replacement index", "invalid syntax",
            "unexpected keyword", "positional arg", "missing required",
            "object has no attribute", "cannot import", "No module named",
            "maximum recursion", "division by zero",
        ]

        error_predictions = []
        for pred in predictions:
            if any(pat in pred for pat in error_patterns):
                error_predictions.append(pred)

        # Only trigger if ALL predictions are errors (not partial failures)
        if len(error_predictions) == len(predictions):
            from collections import Counter
            error_counts = Counter(error_predictions)
            most_common_error = error_counts.most_common(1)[0][0]
            return most_common_error

        # Fallback: if ALL predictions are identical and very short, likely a
        # uniform error (e.g., KeyError('input') becomes the string "'input'")
        from collections import Counter
        pred_counts = Counter(predictions)
        if len(pred_counts) == 1:
            the_pred = pred_counts.most_common(1)[0][0]
            if len(the_pred) < 100:
                return f"All {len(predictions)} queries returned identical short output: {the_pred!r} (likely a runtime error)"

        return ""

    async def _attempt_autofix(self, response: dict, graph_path: str, directory: str,
                                validation_n: int, data: list, error_msg: str) -> float:
        """Attempt to auto-fix a broken workflow and re-evaluate.

        Args:
            response: Original LLM response dict with graph/prompt/modification.
            graph_path: Path to workflows directory.
            directory: Path to current round directory.
            validation_n: Number of validation runs.
            data: Results list to append to.
            error_msg: Error message to include in the fix prompt.

        Returns:
            New avg_score after fix attempt, or 0.0 if fix fails.
        """
        fix_prompt_code, fix_graph_code = self.graph_utils.read_graph_files(self.round + 1, graph_path)
        fixed_graph_body = self.graph_utils.extract_solve_graph(fix_graph_code)

        fixed_response = await self._fix_graph_error(
            response,
            fixed_graph_body[0] if fixed_graph_body else response["graph"],
            fix_prompt_code,
            error_msg,
        )

        if fixed_response:
            self.graph_utils.write_graph_files(
                directory, fixed_response, self.round + 1, self.dataset, self.optimized_path
            )

            self.graph_utils.invalidate_graph_cache(self.round + 1, graph_path)

            try:
                self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)
                logger.info(f"Auto-fix succeeded. Re-evaluating...")

                # Remove stale entries for this round before re-evaluation
                cur_round = self.round + 1
                stale = [d for d in data if d.get("round") == cur_round]
                for s in stale:
                    data.remove(s)

                avg_score = await self.evaluation_utils.evaluate_graph(
                    self, directory, validation_n, data, initial=False
                )

                # Re-save results to overwrite the file that evaluate_graph just wrote
                result_path = self.data_utils.get_results_file_path(graph_path)
                self.data_utils.save_results(result_path, data)

                return avg_score
            except Exception as e2:
                logger.warning(f"Auto-fix still failed: {e2}. Scoring as 0.")
                return 0.0
        else:
            logger.warning("Auto-fix LLM call failed. Scoring as 0.")
            return 0.0

    def _extract_fields_from_response(self, response: str) -> Dict[str, str]:
        """
        Fallback method to extract fields from raw response text using basic parsing
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            # Try to extract XML tags with regex
            import re
            
            # Initialize result dictionary with default values
            result = {
                "modification": "",
                "graph": "",
                "prompt": ""
            }
            
            # Extract each field with regex
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()
            
            # Verify we have at least some content
            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None
            
            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    async def test(self, rounds=None, is_test=True, run_id=None):
        """Run test evaluation for selected rounds.

        Args:
            rounds: List of round numbers to test.
            is_test: If True, use test split; otherwise use validate split.
            run_id: Optional run identifier for repeated runs. When set,
                    results are saved to a separate subdirectory (e.g.,
                    test_results/run_1/) and summary file (e.g.,
                    results_test_run_1.json), so the same round can be
                    evaluated multiple times without being skipped.
        """
        if rounds is None:
            rounds = list(range(1, 22))

        graph_path = f"{self.root_path}/{self.workflows_dir}"

        # Determine filenames based on run_id
        split_label = "test" if is_test else "validate"
        if run_id is not None:
            results_filename = f"results_{split_label}_run_{run_id}.json"
        else:
            results_filename = f"results_{split_label}.json"
        json_file_path = os.path.join(graph_path, results_filename)

        # Load existing test results if any
        data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                data = json.load(f)

        # Skip rounds that have already been tested
        tested_rounds = {d["round"] for d in data}
        rounds_to_run = [r for r in rounds if r not in tested_rounds]
        if rounds_to_run != rounds:
            skipped = set(rounds) - set(rounds_to_run)
            logger.info(f"Skipping already tested rounds: {sorted(skipped)}")

        for round_num in rounds_to_run:
            if run_id is not None:
                subdir = os.path.join(f"{split_label}_results", f"run_{run_id}")
            else:
                subdir = f"{split_label}_results"
            directory = os.path.join(graph_path, f"round_{round_num}", subdir)
            os.makedirs(directory, exist_ok=True)

            self.graph = self.graph_utils.load_graph(round_num, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=is_test)

            new_data = self.data_utils.create_result_data(round_num, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)