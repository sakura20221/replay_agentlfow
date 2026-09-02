import re
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MMLUProBenchmark(BaseBenchmark):
    OPTION_LABELS = "ABCDEFGHIJ"

    # Per-problem extraction prompt is built dynamically based on option count
    EXTRACTION_PROMPT = None  # Disable base class auto-extraction; we handle it in evaluate_problem

    def __init__(self, name: str, file_path: str, log_path: str, llm_config=None):
        super().__init__(name, file_path, log_path, llm_config)

    @staticmethod
    def format_options(options: List[str]) -> str:
        """Format options list into labeled string (A. xxx, B. xxx, ...)."""
        lines = []
        for i, opt in enumerate(options):
            lines.append(f"{MMLUProBenchmark.OPTION_LABELS[i]}. {opt}")
        return "\n".join(lines)

    @staticmethod
    def _valid_letters(num_options: int) -> str:
        """Return valid option letters for the given number of options."""
        return MMLUProBenchmark.OPTION_LABELS[:num_options]

    @staticmethod
    def extract_option_letter(text: str, num_options: int = 10) -> str:
        """Extract option letter from model output, constrained to valid options."""
        valid = MMLUProBenchmark._valid_letters(num_options)
        charset = f"[{valid}]"
        text = text.strip().strip("'\"")
        # Direct single letter
        if len(text) == 1 and text.upper() in valid:
            return text.upper()
        # Prioritized patterns: more explicit first
        patterns = [
            # "answer is X", "answer: X", "Answer: (X)"
            rf"(?:answer|choice)\s*(?:is|:)\s*\(?({charset})\)?",
            # "option X", "option is X", "select X"
            rf"(?:option|select)\s+(?:is\s+)?({charset})\b",
            # Leading "(X)", "X.", "X)" at start of text
            rf"^\(?({charset})\)?[\.\):\s]",
            # Trailing letter at end: "... so B"
            rf"\b({charset})\s*[.\s]*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        # Last resort: find any standalone valid option letter (exclude common words I/A)
        ambiguous = {"A", "I"}
        safe_chars = [c for c in valid if c not in ambiguous]
        if safe_chars:
            safe_set = f"[{''.join(safe_chars)}]"
            matches = re.findall(rf"\b({safe_set})\b", text)
            if matches:
                return matches[-1].upper()
        # If only ambiguous letters remain, return last one
        matches = re.findall(rf"\b({charset})\b", text)
        if matches:
            return matches[-1].upper()
        return text.strip()

    def calculate_score(self, ground_truth: str, prediction: str, num_options: int = 10) -> Tuple[float, str]:
        """Exact match on option letter."""
        pred_letter = self.extract_option_letter(prediction, num_options)
        gt_letter = ground_truth.strip().upper()
        score = 1.0 if pred_letter == gt_letter else 0.0
        return score, pred_letter

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def _extract_answer_for_problem(self, question: str, raw_output: str, num_options: int) -> str:
        """LLM extraction with dynamic prompt based on option count."""
        if self.llm_config is None:
            return None
        try:
            llm = self._get_extraction_llm()
            if llm is None:
                return None
            valid = self._valid_letters(num_options)
            letters_str = ", ".join(valid)
            prompt = (
                f"Given the question: {question}\n"
                f"You need to directly extract the final answer option letter ({letters_str}) "
                f"from the following context. Only output a single uppercase letter, no any other character!\n"
                f"\nContext: {raw_output}"
            )
            return await llm(prompt)
        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to regex: {e}")
            return None

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, str, float, float]:
        question = problem["question"]
        options = problem["options"]
        num_options = len(options)
        expected_output = problem["answer"]  # letter like "A", "B", etc.
        category = problem.get("category", "unknown")

        valid = self._valid_letters(num_options)
        options_str = self.format_options(options)
        inputs = (
            f"The following is a multiple choice question about {category}. "
            f"Please select the correct answer from the given options.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Please output only the option letter ({valid[0]}-{valid[-1]}) as your answer."
        )

        try:
            output, cost = await self._generate_output(graph, inputs)

            # Try LLM extraction first, fall back to regex extraction
            extracted_text = await self._extract_answer_for_problem(question, output, num_options)
            scoring_text = extracted_text if extracted_text is not None else output

            score, extracted_output = self.calculate_score(expected_output, scoring_text, num_options)

            judge_score = await self.llm_judge_answer(question, extracted_output, expected_output)

            if score < 1.0:
                self.log_mismatch(question, expected_output, output, extracted_output)

            return question, output, expected_output, extracted_output, score, cost, judge_score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return question, str(e), expected_output, "", 0.0, 0.0, None

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "extracted_output", "score", "cost", "judge_score"]
