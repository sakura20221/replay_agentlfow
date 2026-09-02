# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of aflow
import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.SHARED_MBPP.workflows.template.operator_an import *
from workspace.SHARED_MBPP.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio

from scripts.utils.code import extract_test_cases_from_jsonl, test_case_2_test_function


from scripts.operators import Operator



class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
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

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, problem, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="SHARED_MBPP", problem=problem)
                
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
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]
        
        result = self.exec_code(problem, solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}

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
