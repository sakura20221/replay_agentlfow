import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import aiofiles
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file


class BaseBenchmark(ABC):
    EXTRACTION_PROMPT = None  # Subclasses override with per-dataset extraction prompt

    def __init__(self, name: str, file_path: str, log_path: str, llm_config=None):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self.llm_config = llm_config
        self._extraction_llm = None

    PASS = "PASS"
    FAIL = "FAIL"

    def _get_extraction_llm(self):
        if self._extraction_llm is None and self.llm_config is not None:
            from scripts.async_llm import create_llm_instance
            self._extraction_llm = create_llm_instance(self.llm_config)
        return self._extraction_llm

    JUDGE_PROMPT = (
        "Given the question: {question}\n"
        "We have the ground truth answer: {expected_answer}\n"
        "Now please judge if the following answer is correct: {predicted_answer}\n"
        "Note: Different formats of the same answer (e.g., '28\\%' vs '28', "
        "'\\text{{Devon}}' vs 'Devon', '\\frac{{1}}{{2}}' vs '0.5') should be considered correct.\n"
        "Only output 1 as correct, or 0 as wrong, no any other character."
    )

    async def llm_extract_answer(self, question: str, raw_output: str) -> Optional[str]:
        if self.EXTRACTION_PROMPT is None or self.llm_config is None:
            return None
        try:
            llm = self._get_extraction_llm()
            if llm is None:
                return None
            prompt = self.EXTRACTION_PROMPT.format(question=question, output=raw_output)
            response = await llm(prompt)
            return response
        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to regex: {e}")
            return None

    async def llm_judge_answer(self, question: str, predicted_answer: str, expected_answer: str) -> Optional[int]:
        """Use LLM as a judge to verify if a predicted answer is correct.
        Returns 1 (correct), 0 (wrong), or None (judge unavailable/failed)."""
        if self.llm_config is None:
            return None
        try:
            llm = self._get_extraction_llm()
            if llm is None:
                return None
            prompt = self.JUDGE_PROMPT.format(
                question=question,
                predicted_answer=predicted_answer,
                expected_answer=expected_answer,
            )
            response = await llm(prompt)
            response = response.strip()
            if response == "1":
                return 1
            elif response == "0":
                return 0
            else:
                logger.warning(f"LLM judge returned unexpected response: {response}")
                return None
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return None

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    @staticmethod
    def _compute_per_problem_costs(records: List[dict]) -> None:
        """Post-process records to derive per-problem cost from cumulative cost.

        The graph returns cumulative cost from a shared usage_tracker. After all
        problems finish, we sort by cumulative cost and difference adjacent values
        to recover per-problem cost. The original cumulative value is kept as
        ``accumulated_cost``; ``cost`` is overwritten with the per-problem value.
        """
        cumulative_costs = [(i, r["cost"]) for i, r in enumerate(records)]
        cumulative_costs.sort(key=lambda x: x[1])

        per_problem = [0.0] * len(records)
        for rank, (orig_idx, cum) in enumerate(cumulative_costs):
            per_problem[orig_idx] = cum if rank == 0 else cum - cumulative_costs[rank - 1][1]

        for i, record in enumerate(records):
            record["accumulated_cost"] = record["cost"]
            record["cost"] = per_problem[i]

    def save_results(self, results: List[Tuple[Any, ...]], columns: List[str]):
        records = [dict(zip(columns, row)) for row in results]

        # Derive per-problem cost from cumulative cost (post-hoc)
        self._compute_per_problem_costs(records)

        scores = [r["score"] for r in records]
        per_costs = [r["cost"] for r in records]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        t_cost = sum(per_costs) if per_costs else 0.0
        a_cost = t_cost / len(records) if records else 0.0

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{avg_score:.5f}_{current_time}"

        # Save .jsonl (one JSON object per line)
        jsonl_file = os.path.join(self.log_path, f"{base_name}.jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Save .json (full array)
        json_file = os.path.join(self.log_path, f"{base_name}.json")
        write_json_file(json_file, records, encoding="utf-8", indent=4)

        logger.info(f"Results saved to {jsonl_file} and {json_file}")
        return avg_score, a_cost, t_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 50):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 50):
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost

