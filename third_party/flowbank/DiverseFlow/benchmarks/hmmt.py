import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class HMMTBenchmark(BaseBenchmark):
    EXTRACTION_PROMPT = (
        "Given the question: {question}\n"
        "You need to directly extract the final answer 'x' and output in the format "
        "'\\boxed{{x}}' (without any modification!) for this question from the following "
        "context, no any other character! Context: {output}"
    )

    def __init__(self, name: str, file_path: str, log_path: str, llm_config=None):
        super().__init__(name, file_path, log_path, llm_config)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                pred_val = self.parse_digits(prediction)
                ref_val = self.parse_digits(reference)
                if isclose(pred_val, ref_val, abs_tol=1e-3):
                    return True
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
            if num.endswith("\\%"):
                num = num[:-2]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    @staticmethod
    def _normalize_latex(s: str) -> str:
        """Convert common LaTeX notation to sympy-parseable expressions."""
        s = s.strip()
        s = re.sub(r'\\dfrac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
        s = s.replace('\\%', '')
        s = s.replace('\\cdot', '*')
        s = s.replace('\\times', '*')
        s = s.replace('\\left', '').replace('\\right', '')
        s = s.replace('\\pi', 'pi')
        s = re.sub(r'\\text\{[^{}]*\}', '', s)
        s = re.sub(r'\\[,;!:]', '', s)
        s = s.strip()
        return s

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            normalized = self._normalize_latex(s)
            if normalized != s:
                try:
                    return parse_expr(normalized)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            output, cost = await self._generate_output(graph, input_text)

            # Try LLM extraction first, fall back to raw output
            extracted_text = await self.llm_extract_answer(input_text, output)
            scoring_text = extracted_text if extracted_text is not None else output

            uni_score, extracted_output = self.calculate_score(expected_output, scoring_text)

            # LLM Judge fallback: when rule-based scoring gives 0, use LLM to double-check
            if uni_score == 0:
                expected_answer = self.extract_model_answer(expected_output)
                judge_result = await self.llm_judge_answer(input_text, extracted_output, expected_answer)
                if judge_result == 1:
                    logger.info(f"LLM Judge overrode rule-based score: '{extracted_output}' vs '{expected_answer}'")
                    uni_score = 1
                else:
                    self.log_mismatch(
                        input_text,
                        expected_output,
                        output,
                        extracted_output,
                        extract_answer_code=self.get_function_code(self.extract_model_answer),
                    )

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
