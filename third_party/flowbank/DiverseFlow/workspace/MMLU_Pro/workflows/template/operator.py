# -*- coding: utf-8 -*-
# Operator module - imports from scripts.operators
# MMLU-Pro uses a custom ScEnsemble with numeric labels to avoid
# collision with the A-J answer options in multiple-choice questions.
from scripts.operators import *

from typing import List
from pydantic import BaseModel, Field
from scripts.operators import Operator, ScEnsemble as _OriginalScEnsemble
from scripts.async_llm import AsyncLLM

SC_ENSEMBLE_NUMERIC_PROMPT = """Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_number" field, output ONLY the numeric ID (1, 2, 3, etc.) corresponding to the most consistent solution. Do NOT output a letter — output the number only.
"""


class ScEnsembleNumericOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_number: str = Field(default="", description="The number of the most consistent solution (1, 2, 3, etc.).")


class ScEnsemble(_OriginalScEnsemble):
    """MMLU-Pro specific ScEnsemble that uses numeric labels (1, 2, 3...)
    instead of letter labels (A, B, C...) to avoid conflict with
    multiple-choice answer options A-J."""

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            num_label = str(index + 1)
            answer_mapping[num_label] = index
            solution_text += f"{num_label}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_NUMERIC_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleNumericOp, prompt, mode="xml_fill")

        answer = response.get("solution_number", "").strip()

        if answer in answer_mapping:
            return solutions[answer_mapping[answer]]

        # Fallback: try to extract a number from the response
        import re
        nums = re.findall(r'\d+', answer)
        for n in nums:
            if n in answer_mapping:
                return solutions[answer_mapping[n]]

        # Last resort: return first solution
        return solutions[0]
