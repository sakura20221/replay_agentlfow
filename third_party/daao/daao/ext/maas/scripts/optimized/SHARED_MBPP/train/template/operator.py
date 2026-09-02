import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple

from daao.ext.maas.scripts.optimized.HumanEval.train.template.operator_an import *
from daao.ext.maas.scripts.optimized.HumanEval.train.template.op_prompt import *
from daao.ext.maas.scripts.utils import extract_test_cases_from_jsonl, test_case_2_test_function
from daao.actions.action_node import ActionNode
from daao.llm import LLM
from daao.logs import logger
import re


class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

class CustomCodeGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response

class Generate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response
    
class GenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response

class MultiGenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        
        response1 = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        response2 = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        response3 = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        
        return {"response": [response1, response2, response3]}
    
class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = _shim_safe_format(SC_ENSEMBLE_PROMPT, problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

class Test(Operator):
    def __init__(self, llm: LLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, problem, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset=_shim_code_dataset(), problem=problem)
                
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                _shim_exec_snippet(test_code)
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 3
    ):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(problem, solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = _shim_safe_format(REFLECTION_ON_PUBLIC_TEST_PROMPT, 
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["reflection_and_solution"]
            else:
                prompt = _shim_safe_format(REFLECTION_ON_PUBLIC_TEST_PROMPT, 
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["reflection_and_solution"]
        
        result = self.exec_code(problem, solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}
        
class SelfRefine(Operator):
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution):
        prompt = _shim_safe_format(SELFREFINE_PROMPT, problem=problem, solution=solution)
        response = await self._fill_node(SelfRefineOp, prompt, mode="code_fill")
        return response
    
# class EarlyStop(Operator):
#     def __init__(self, llm: LLM, name: str = "EarlyStop"):
#         super().__init__(llm, name)

#     async def __call__(self):
#         return NotImplementedError


# --- shared-layer shim: brace-safe prompt substitution ---
# The shipped templates embed LaTeX demonstration examples such as \boxed{-2},
# \frac{1}{3} and \boxed{144\pi}. str.format() reads every one of those as a
# replacement field, so GENERATE_COT_PROMPT.format(...) raises KeyError("-2") on
# *any* input -- 40 of 40 MATH problems reproduce it. Two operators (GenerateCoT,
# MultiGenerateCoT) therefore never worked, and MaAS's own broad `except` in
# evaluate_problem masked it as a score of 0.
#
# This substitutes only the named placeholders and leaves every other brace as
# literal text, which is what the templates intend.
def _shim_safe_format(_template, **kwargs):
    out = _template
    for _key, _value in kwargs.items():
        out = out.replace("{" + _key + "}", str(_value))
    return out


# --- shared-layer shim (agent_wf_v2) --- exec guard v1
# See shims/exec_guard_patch.py. Bounds execution of model-written code with a
# process that can actually be killed, and preserves each operator's own
# exception-based classification of the result.
import sys as _shim_eg_sys
from pathlib import Path as _ShimEgPath

_SHIM_EG_DISALLOWED = []
_shim_eg_module = None


def _shim_exec_guard():
    """Import shared/exec_guard.py, locating it by walking up from this file."""
    global _shim_eg_module
    if _shim_eg_module is None:
        root = _ShimEgPath(__file__).resolve()
        for _ in range(12):
            root = root.parent
            if (root / "shared" / "exec_guard.py").exists():
                break
        else:
            raise RuntimeError("shared/exec_guard.py not found above " + __file__)
        if str(root / "shared") not in _shim_eg_sys.path:
            _shim_eg_sys.path.insert(0, str(root / "shared"))
        import exec_guard as _loaded

        _shim_eg_module = _loaded
    return _shim_eg_module


def _shim_run_solve(code, timeout=30):
    """The author's prohibited-import refusal, then guarded execution.

    The list and the message are lifted from this module's own `run_code` at
    install time (see shims/exec_guard_patch.py), because bypassing that check
    would change what the operator refuses to run -- matplotlib and friends are
    rejected by design, and a plot attempt inside a sandbox would fail in a
    different way instead of being refused.
    """
    for _lib in _SHIM_EG_DISALLOWED:
        if f"import {_lib}" in code or f"from {_lib}" in code:
            try:
                logger.info("Detected prohibited import: %s", _lib)
            except Exception:  # noqa: BLE001 - logging must not decide the outcome
                pass
            return "Error", f"Prohibited import: {_lib} and graphing functionalities"
    return _shim_exec_guard().run_solve(code, timeout)


def _shim_exec_snippet(test_code, timeout=30):
    """Run one snippet; raise what the caller's except-branches expect.

    AssertionError is re-raised as AssertionError because every Test operator
    branches on it to record a failed test case, separately from a crash. A
    timeout arrives as TimeoutError, which is an Exception, so it lands in the
    operators' generic branch and is recorded as a failure rather than killing the
    run.
    """
    outcome = _shim_exec_guard().run_snippets([test_code], timeout=timeout)[0]
    if outcome["status"] == "ok":
        return
    kind = outcome.get("error_type") or "RuntimeError"
    message = outcome.get("message", "")
    if kind == "AssertionError":
        raise AssertionError(message)
    if kind == "TimeoutError":
        raise TimeoutError(message)
    raise RuntimeError(f"{kind}: {message}")


# --- shared-layer shim (agent_wf_v2) --- scensemble v2
# ScEnsemble label-space fix, backported verbatim in behaviour from FlowBank's
# own MMLU-Pro operator (author commit dde948d). See shims/maas_family/install.py
# for why this is applied to MaAS/DAAO as well.
import os as _shim_os
import re as _shim_re
import sys as _shim_sys
from pydantic import BaseModel as _ShimBaseModel, Field as _ShimField

SC_ENSEMBLE_NUMERIC_PROMPT = """Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_number" field, output ONLY the numeric ID (1, 2, 3, etc.) corresponding to the most consistent solution. Do NOT output a letter - output the number only.
"""


class ScEnsembleNumericOp(_ShimBaseModel):
    thought: str = _ShimField(default="", description="The thought of the most consistent solution.")
    solution_number: str = _ShimField(default="", description="The number of the most consistent solution (1, 2, 3, etc.).")


def _shim_code_dataset() -> str:
    """Which code benchmark's public tests the Test operator should look up.

    The seeded SHARED_MBPP workspace imports this template from the authors'
    HumanEval workspace -- that is how MaAS lays out workspaces, and it is why
    patching SHARED_MBPP's own copy changed nothing: that copy is never imported.
    The hardcoded `dataset=_shim_code_dataset()` therefore sent MBPP function names to
    humaneval_public_test.jsonl, which MaAS does not ship at all (1,020 samples
    discarded) and which DAAO ships without those names (255 discarded).

    Decided at runtime for the same reason as the label space below: one template
    serves several datasets, so the answer cannot be baked in at install time.
    """
    marker = (_shim_os.getenv("SHIM_DATASET", "") or " ".join(_shim_sys.argv)).upper()
    return "SHARED_MBPP" if "MBPP" in marker else "HumanEval"


def _shim_on_mmlu_pro() -> bool:
    """Whether this process is running the dataset whose answers are letters.

    Read from the environment first (sweep.py sets SHIM_DATASET per job) and from
    argv second, so a manual `optimize.py --dataset SHARED_MMLUPRO` outside the
    sweep still gets the fix instead of silently reverting to letters.
    """
    marker = _shim_os.getenv("SHIM_DATASET", "") or " ".join(_shim_sys.argv)
    marker = marker.upper()
    return "MMLUPRO" in marker or "MMLU_PRO" in marker


class ScEnsemble(ScEnsemble):  # noqa: F811 - wraps the author class defined above
    async def __call__(self, solutions, problem: str):
        numeric = _shim_on_mmlu_pro()
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            label = str(index + 1) if numeric else chr(65 + index)
            answer_mapping[label] = index
            solution_text += f"{label}: \n{str(solution)}\n\n\n"

        if numeric:
            prompt = _shim_safe_format(SC_ENSEMBLE_NUMERIC_PROMPT,
                                       question=problem, solutions=solution_text)
            response = await self._fill_node(ScEnsembleNumericOp, prompt, mode="xml_fill")
            answer = str(response.get("solution_number", "")).strip()
        else:
            prompt = _shim_safe_format(SC_ENSEMBLE_PROMPT,
                                       problem=problem, solutions=solution_text)
            response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")
            answer = str(response.get("solution_letter", "")).strip().upper()

        if answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}

        # Fallback ladder, as in the FlowBank original: pull a label out of a
        # wordier reply, then fall back to the first solution. Reaching the last
        # step is logged rather than silent, because it is a real degradation of
        # the ensemble -- the selection stops being a choice.
        for token in _shim_re.findall(r"\d+" if numeric else r"[A-Z]", answer):
            if token in answer_mapping:
                return {"response": solutions[answer_mapping[token]]}
        _shim_sys.stderr.write(
            f"[shared_shim] ScEnsemble got {answer!r}, not one of "
            f"{sorted(answer_mapping)}; falling back to the first solution\n")
        return {"response": solutions[0]}
