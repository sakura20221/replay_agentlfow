"""Make every repo's generated-code execution killable, identically.

Four repos execute code the model wrote, and three of them cannot stop it:

* MaAS and DAAO: `Programmer.exec_code` opens `ProcessPoolExecutor` in a `with`
  block and, on timeout, calls `shutdown(wait=False, cancel_futures=True)`. That
  does not kill a child already running, and the `with` exit then calls
  `shutdown(wait=True)` -- which waits for it forever. Nominal timeout 600 s;
  actual behaviour on an infinite loop, a permanently hung job.
* AFlow: the same code with `timeout=30`, so the same permanent hang, plus
  `Test.exec_code` running `exec(test_code, globals())` with no timeout at all.
* FlowBank: already survives this. Its Test keeps a long-lived pool outside any
  `with`, times out per case and rebuilds the pool, so a hang costs a leaked
  spinning process but not the job. Left alone apart from that leak.

The fix is applied identically to every method rather than per repo, for the same
reason the answer format lives in one place: a timeout policy that differs by
method would show up in the results as a difference between methods.

What changes is only *how* execution is bounded. The operators' control flow is
untouched -- the guarded helper re-raises `AssertionError` for a failed assert and
a plain exception otherwise, so each author's own except-branches still classify
the outcome the way they did.
"""

from __future__ import annotations

import re
from pathlib import Path

MARKER = "# --- shared-layer shim (agent_wf_v2) --- exec guard v1"

# Appended to any operator module that executes generated code. The import walks
# up to find shared/exec_guard.py rather than relying on PYTHONPATH: these modules
# are imported from inside four different repos, each with its own cwd and venv.
HELPER = '''

# --- shared-layer shim (agent_wf_v2) --- exec guard v1
# See shims/exec_guard_patch.py. Bounds execution of model-written code with a
# process that can actually be killed, and preserves each operator's own
# exception-based classification of the result.
import sys as _shim_eg_sys
from pathlib import Path as _ShimEgPath

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
'''

# Replaces the whole body of Programmer.exec_code. The signature keeps its name and
# its `timeout` keyword so any caller passing one still works; the default drops
# from 600 (MaAS/DAAO) to 30, which is ten times the slowest correct solution
# measured across the recorded runs (3.16 s).
EXEC_CODE_REPLACEMENT = '''    async def exec_code(self, code, timeout=30):
        """Execute generated code under a timeout that can be enforced.

        Replaced by shims/exec_guard_patch.py. The original opened a
        ProcessPoolExecutor in a `with` block, and on timeout cancelled without
        killing -- so the block's exit waited on the runaway child forever. Returns
        the same ("Success"|"Error", text) pair the operator's callers expect.
        """
        import asyncio as _shim_eg_asyncio

        loop = _shim_eg_asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _shim_run_solve(code, timeout))
'''

# The author's own refusal list, as written in run_code. Matched loosely on
# whitespace because the four repos indent it differently.
_DISALLOWED_RE = re.compile(r"disallowed_imports\s*=\s*\[(.*?)\]", re.DOTALL)


def _disallowed_list(text: str) -> str:
    """The literal list source from this file's run_code, or a safe default."""
    match = _DISALLOWED_RE.search(text)
    if match:
        return "[" + match.group(1) + "]"
    # No run_code in this module (a Test-only operator file). An empty list means
    # "refuse nothing", which is correct: there was no refusal here to preserve.
    return "[]"


def patch_file(path: Path) -> str:
    """Patch one operator module. Returns 'patched', 'already' or 'skipped'."""
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already"

    disallowed = _disallowed_list(text)
    changed = False

    # 1. Programmer.exec_code -- replace from its `async def` to the next method.
    start = text.find("    async def exec_code(self, code")
    if start != -1:
        rest = text[start:]
        # The next sibling method at the same indentation ends the body. Anchored on
        # a newline so a nested `async def` inside the body could not end it early.
        match = re.search(r"\n    (?:async )?def ", rest[1:])
        end = start + 1 + match.start() + 1 if match else len(text)
        text = text[:start] + EXEC_CODE_REPLACEMENT + text[end:]
        changed = True

    # 2. Test.exec_code -- the bare exec of a generated test case.
    if "exec(test_code, globals())" in text:
        text = text.replace("exec(test_code, globals())", "_shim_exec_snippet(test_code)")
        changed = True
    if "exec(test_code, {})" in text:
        text = text.replace("exec(test_code, {})", "_shim_exec_snippet(test_code)")
        changed = True

    if not changed:
        return "skipped"
    helper = HELPER.replace(
        "_shim_eg_module = None",
        f"_SHIM_EG_DISALLOWED = {disallowed}\n_shim_eg_module = None", 1)
    path.write_text(text + helper, encoding="utf-8")
    return "patched"


def verify_file(path: Path) -> list[str]:
    """Problems with one patched file, as human-readable strings."""
    text = path.read_text(encoding="utf-8")
    problems = []
    if MARKER not in text:
        problems.append(f"{path.name}: guard not installed")
        return problems
    if "ProcessPoolExecutor(max_workers=1) as executor" in text:
        problems.append(f"{path.name}: still opens a pool in a with-block")
    if "timeout=600" in text:
        problems.append(f"{path.name}: still carries the 600s timeout")
    for pattern in ("exec(test_code, globals())", "exec(test_code, {})"):
        if pattern in text:
            problems.append(f"{path.name}: unguarded {pattern}")
    # The author's refusal list must have survived into the helper. Losing it would
    # let matplotlib code run where the operator meant to reject it, which is a
    # behaviour change hiding inside an infrastructure fix.
    author_list = _DISALLOWED_RE.search(text)
    if author_list:
        libraries = re.findall(r'["\']([A-Za-z0-9_]+)["\']', author_list.group(1))
        injected = re.search(r"_SHIM_EG_DISALLOWED\s*=\s*\[(.*?)\]", text, re.DOTALL)
        kept = re.findall(r'["\']([A-Za-z0-9_]+)["\']', injected.group(1)) if injected else []
        missing = sorted(set(libraries) - set(kept))
        if missing:
            problems.append(f"{path.name}: refusal list lost {missing}")
    return problems
