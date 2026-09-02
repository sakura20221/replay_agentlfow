import concurrent
import sys
import traceback
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.SHARED_AMC.workflows.template.operator_an import *
from workspace.SHARED_AMC.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio


from scripts.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code, timeout=30):
        """Execute generated code under a timeout that can be enforced.

        Replaced by shims/exec_guard_patch.py. The original opened a
        ProcessPoolExecutor in a `with` block, and on timeout cancelled without
        killing -- so the block's exit waited on the runaway child forever. Returns
        the same ("Success"|"Error", text) pair the operator's callers expect.
        """
        import asyncio as _shim_eg_asyncio

        loop = _shim_eg_asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _shim_run_solve(code, timeout))
    async def code_generate(self, problem, analysis, feedback, mode):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


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

# --- shared-layer shim (agent_wf_v2) --- exec guard v1
# See shims/exec_guard_patch.py. Bounds execution of model-written code with a
# process that can actually be killed, and preserves each operator's own
# exception-based classification of the result.
import sys as _shim_eg_sys
from pathlib import Path as _ShimEgPath

_SHIM_EG_DISALLOWED = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]
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
