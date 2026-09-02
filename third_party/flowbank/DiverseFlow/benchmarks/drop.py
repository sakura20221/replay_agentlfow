import re
import string
from collections import Counter
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class DROPBenchmark(BaseBenchmark):
    EXTRACTION_PROMPT = (
        "Given the question: {question}\n"
        "You need to directly extract the final answer (without any modification!) "
        "for this question from the following context, no any other character! "
        "The final answer should be concise, accurate, and directly addressing the question. "
        "For example, if the answer is a number, just provide the number. "
        "If the answer is Yes/No, just response Yes/No, no any other character. "
        "\nContext: {output}"
    )

    def __init__(self, name: str, file_path: str, log_path: str, llm_config=None):
        super().__init__(name, file_path, log_path, llm_config)

    def normalize_answer(self, s: str) -> List[str]:
        """
        Normalize answers for evaluation.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["context"]
        expected_output = problem["ref_text"]
        answers = expected_output.split("|")

        try:
            output, cost = await self._generate_output(graph, input_text)

            # Try LLM extraction first, fall back to raw output
            extracted_text = await self.llm_extract_answer(input_text, output)
            scoring_text = extracted_text if extracted_text is not None else output

            # Always compute F1 score
            f1_scores = []
            for answer in answers:
                if answer.strip() != "":
                    output_parts = scoring_text.split("|")
                    for output_part in output_parts:
                        f1_val, _ = self.calculate_score(answer, output_part)
                        f1_scores.append(f1_val)
            uni_score = max(f1_scores) if f1_scores else 0.0

            # Compute LLM judge score
            answer_list = ", or ".join(a for a in answers if a.strip())
            judge_score = await self.llm_judge_answer(input_text, scoring_text, answer_list)

            if uni_score < 0.3:
                self.log_mismatch(input_text, expected_output, output, output)

            return input_text, output, expected_output, uni_score, cost, judge_score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, None

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost", "judge_score"]
