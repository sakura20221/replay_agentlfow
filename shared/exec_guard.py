"""Run model-generated Python under a timeout that is actually enforceable.

Three mechanisms were available and two of them do not work here:

* a thread (`ThreadPoolExecutor` + `future.result(timeout=...)`) cannot be
  stopped. `while True: pass` in generated code keeps a core busy for the rest of
  the run, and the pool's shutdown blocks on it.
* `signal.alarm` only fires between bytecode instructions, so it interrupts a
  Python-level loop but not a C-level one -- `itertools.count()` fed to `sum`, or a
  regex with catastrophic backtracking, ignores it entirely.
* a separate OS process can be killed unconditionally, which is why that is what
  this module does.

The child is a plain `sys.executable -c` interpreter fed on stdin, not a fork of
the caller. Forking an asyncio process that has threads copies whatever locks
those threads hold -- the operator modules log from inside the executed code path,
so a child that inherits a held logging lock deadlocks, and a deadlocked child is
the failure this module exists to prevent. A fresh interpreter also cannot see the
parent's imports, which is the isolation the caller wants anyway. It costs about
40 ms of startup; measured execution times for correct solutions are a median of
0.4 ms and a worst case of 3.16 s, so the overhead is real but small, and the
snippets belonging to one solution are batched into a single child to pay it once.

The observed failure this replaces: MaAS's Programmer wrapped a
`ProcessPoolExecutor` in a `with` block and cancelled on timeout with
`shutdown(wait=False, cancel_futures=True)`. That does not kill a child that is
already running, and the `with` block's exit then calls `shutdown(wait=True)`,
which waits for it forever. So generated code containing an infinite loop did not
time out after the nominal 600 s -- it hung the job permanently.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

# 30 s, from measurement rather than taste: across the recorded MBPP and MATH
# runs the slowest *correct* solution took 3.16 s, so 30 s is an order of
# magnitude of headroom and still bounded. The 600 s the operator asked for is
# longer than many whole jobs' useful work.
DEFAULT_TIMEOUT = 30.0

# Runs inside the child. Kept as source rather than a module so the child needs no
# import path and no package on sys.path.
_CHILD = r"""
import contextlib, io, json, os, sys, traceback

request = json.loads(sys.stdin.read())
mode = request["mode"]
timeout_each = request.get("per_snippet_timeout")
# Results go to a file, not to stdout. Generated solutions print -- MBPP's do it
# constantly -- and a snippet that printed 100 kB interleaved its output with the
# JSON payload, so every result in the batch came back unparseable and was counted
# as a failure. The file is opened by path given by the parent, so nothing the
# generated code writes to fd 1 or 2 can reach it.
result_path = request["result_path"]

# Per-snippet alarm inside the child, in addition to the parent's hard kill. The
# parent's kill is what guarantees termination; this one exists so that ONE hanging
# snippet does not cost the results of the snippets after it. It is a best effort:
# a C-level loop ignores it and the parent then kills the whole child.
if timeout_each:
    import signal

    def _expire(signum, frame):
        raise TimeoutError("snippet exceeded its share of the budget")

    try:
        signal.signal(signal.SIGALRM, _expire)
    except (ValueError, AttributeError):
        timeout_each = None

results = []
for snippet in request["snippets"]:
    if timeout_each:
        signal.setitimer(signal.ITIMER_REAL, timeout_each)
    namespace = {"__name__": "__generated__"}
    try:
        # The snippet's own output is discarded rather than captured: the caller
        # grades the returned value or the assertions, never the printing, and a
        # solution that prints in a loop would otherwise fill memory before the
        # timeout fired.
        sink = open(os.devnull, "w")
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(snippet, "<generated>", "exec"), namespace)
            outcome = {"status": "ok", "value": ""}
            if mode == "call_entry":
                entry = request.get("entry", "solve")
                target = namespace.get(entry)
                if callable(target):
                    outcome["value"] = str(target())
                else:
                    outcome = {"status": "error", "error_type": "NameError",
                               "message": "Function '%s' not found" % entry,
                               "traceback": ""}
    except AssertionError as exc:
        outcome = {"status": "error", "error_type": "AssertionError",
                   "message": str(exc), "traceback": "".join(
                       traceback.format_exception(*sys.exc_info()))}
    except BaseException as exc:  # includes the alarm's TimeoutError
        outcome = {"status": "error", "error_type": type(exc).__name__,
                   "message": str(exc), "traceback": "".join(
                       traceback.format_exception(*sys.exc_info()))}
    finally:
        if timeout_each:
            signal.setitimer(signal.ITIMER_REAL, 0)
    results.append(outcome)
    # Rewritten after every snippet, so a child killed mid-batch still leaves the
    # verdicts it had already reached.
    with open(result_path, "w") as handle:
        json.dump(results, handle)
"""


def _timeout_result(message: str) -> dict:
    return {"status": "error", "error_type": "TimeoutError",
            "message": message, "traceback": ""}


def run_snippets(snippets: list[str], timeout: float = DEFAULT_TIMEOUT,
                 mode: str = "exec", entry: str = "solve") -> list[dict]:
    """Execute each snippet in one throwaway interpreter; never raise.

    Returns one result dict per snippet, in order. Snippets that the child never
    reached -- because it was killed, or crashed outright -- come back as timeouts
    rather than silently missing, so a caller counting failures cannot mistake a
    killed child for a clean pass.
    """
    if not snippets:
        return []
    handle = tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False,
                                         prefix="exec_guard_")
    handle.write("[]")
    handle.close()
    result_path = handle.name
    payload = {
        "snippets": list(snippets),
        "mode": mode,
        "entry": entry,
        "result_path": result_path,
        # Divided so the whole batch still fits the budget, with a floor: a
        # single-snippet batch gets the full timeout.
        "per_snippet_timeout": max(timeout / max(len(snippets), 1), 1.0),
    }
    stderr = ""
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child on timeout, and waits for it, so nothing
        # is left spinning on a machine that is shared with other people's jobs.
        timed_out = True
    except (OSError, ValueError) as exc:
        _unlink(result_path)
        return [{"status": "error", "error_type": type(exc).__name__,
                 "message": f"could not start the sandbox interpreter: {exc}",
                 "traceback": ""} for _ in snippets]

    # Read whatever the child managed to record, even when it was killed: the
    # snippets it finished before hanging keep their verdicts.
    try:
        with open(result_path, "r", encoding="utf-8") as reader:
            results = json.load(reader)
    except (OSError, json.JSONDecodeError):
        results = []
    finally:
        _unlink(result_path)
    if not isinstance(results, list):
        results = []

    # Fewer results than snippets means the child was killed or died -- segfault,
    # MemoryError, os._exit() inside generated code.
    while len(results) < len(snippets):
        if timed_out:
            results.append(_timeout_result(
                f"Code execution timed out after {timeout:g}s"))
        else:
            detail = stderr.strip()[-300:] or "child exited early"
            results.append({"status": "error", "error_type": "ChildDied",
                            "message": detail, "traceback": ""})
    return results[: len(snippets)]


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def run_solve(code: str, timeout: float = DEFAULT_TIMEOUT,
              entry: str = "solve") -> tuple[str, str]:
    """MaAS/DAAO Programmer contract: ("Success", str(result)) or ("Error", why)."""
    result = run_snippets([code], timeout=timeout, mode="call_entry", entry=entry)[0]
    if result["status"] == "ok":
        return "Success", result["value"]
    message = result.get("message", "")
    if result.get("error_type") == "TimeoutError":
        return "Error", "Code execution timed out"
    if result.get("error_type") == "NameError" and "not found" in message:
        return "Error", message
    return "Error", f"Execution error: {message}\n{result.get('traceback', '')}"
