import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.SHARED_MMLUPRO.workflows.template.operator_an import *
from workspace.SHARED_MMLUPRO.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re


from scripts.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response

class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

# --- shared-layer shim (agent_wf_v2) --- scensemble v2
# MMLU-Pro label-space fix. The author operator above labels candidate solutions
# A, B, C and asks which letter is best; MMLU-Pro's own options are (A)-(J), so a
# reply of "E" means the question's option E and indexes a candidate that does not
# exist -- KeyError, and the sample is lost rather than merely scored wrong.
# Numeric labels remove the ambiguity. Lifted from FlowBank's operator for the
# same dataset; the author class above is left untouched and shadowed.
import re as _shim_re
import sys as _shim_sys
from pydantic import BaseModel as _ShimBaseModel, Field as _ShimField

# The question is optional, because the template's own SC_ENSEMBLE_PROMPT does not
# include it either -- this workspace's ensemble judges the candidates alone. Kept
# as a leading slot so a graph that does pass the problem loses nothing.
SC_ENSEMBLE_NUMERIC_PROMPT = """{question}Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_number" field, output ONLY the numeric ID (1, 2, 3, etc.) corresponding to the most consistent solution. Do NOT output a letter - output the number only.
"""


class ScEnsembleNumericOp(_ShimBaseModel):
    thought: str = _ShimField(default="", description="The thought of the most consistent solution.")
    solution_number: str = _ShimField(default="", description="The number of the most consistent solution (1, 2, 3, etc.).")


class ScEnsemble(ScEnsemble):  # noqa: F811 - shadows the author class above
    # Default, not required: see the note in shims/aflow/install.py. The class this
    # shadows takes `solutions` alone, and the optimiser generates calls to match.
    async def __call__(self, solutions: List[str], problem: str = ""):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            label = str(index + 1)
            answer_mapping[label] = index
            solution_text += f"{label}: \n{str(solution)}\n\n\n"

        question = f"Given the question described as follows: {problem}\n" if problem else ""
        prompt = SC_ENSEMBLE_NUMERIC_PROMPT.format(question=question, solutions=solution_text)
        response = await self._fill_node(ScEnsembleNumericOp, prompt, mode="xml_fill")
        answer = str(response.get("solution_number", "")).strip()

        if answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}
        for token in _shim_re.findall(r"\d+", answer):
            if token in answer_mapping:
                return {"response": solutions[answer_mapping[token]]}
        _shim_sys.stderr.write(
            f"[shared_shim] ScEnsemble got {answer!r}, not one of "
            f"{sorted(answer_mapping)}; falling back to the first solution\n")
        return {"response": solutions[0]}
