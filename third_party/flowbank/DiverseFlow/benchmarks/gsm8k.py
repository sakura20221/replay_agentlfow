# -*- coding: utf-8 -*-
# @Desc    : test on gsm8k
import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class GSM8KBenchmark(BaseBenchmark):
    EXTRACTION_PROMPT = (
        "Given the question: {question}\n"
        "You need to directly extract the numerical final answer (without any modification!) "
        "for this question from the following context, no any other character! Context: {output}"
    )

    def __init__(self, name: str, file_path: str, log_path: str, llm_config=None):
        super().__init__(name, file_path, log_path, llm_config)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, float, float, float]:
        input_text = problem["question"]
        expected_output = self.extract_number(problem["answer"])

        try:
            output, cost = await self._generate_output(graph, input_text)

            # Try LLM extraction first, fall back to raw output
            extracted_text = await self.llm_extract_answer(input_text, output)
            scoring_text = extracted_text if extracted_text is not None else output

            predicted_number = self.extract_number(scoring_text)
            score, extracted_output = self.calculate_score(expected_output, predicted_number)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
