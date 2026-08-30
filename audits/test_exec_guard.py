#!/usr/bin/env python3
"""Does the guard survive the things that actually hung the jobs?

Each case below is a shape taken from real generated code, not an invented one:
the Python-level infinite loop and the C-level one are what `find_hanging_solution`
and `test_generated_code_hangs` found in the recorded MBPP solutions, and the
"passes then hangs" case is the one that matters most -- it proves a hanging
snippet does not take the earlier results down with it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import exec_guard  # noqa: E402

failures = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global failures
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures += 1


# 1. A correct solution still works, and returns its value.
status, value = exec_guard.run_solve("def solve():\n    return 6 * 7\n", timeout=10)
check(status == "Success" and value == "42", "correct solve() returns its value",
      f"{status}/{value!r}")

# 2. A Python-level infinite loop is killed, and quickly.
started = time.monotonic()
status, value = exec_guard.run_solve("def solve():\n    while True:\n        pass\n", timeout=3)
elapsed = time.monotonic() - started
check(status == "Error" and "timed out" in value, "python-level infinite loop times out",
      f"{value!r}")
check(elapsed < 8, "and does so near the deadline", f"{elapsed:.1f}s for a 3s budget")

# 3. A C-level loop, which signal.alarm cannot interrupt. This is the case that
#    made signal-based timeouts useless: the interpreter never regains control.
started = time.monotonic()
status, value = exec_guard.run_solve(
    "import itertools\n"
    "def solve():\n"
    "    return sum(1 for _ in itertools.count())\n", timeout=3)
elapsed = time.monotonic() - started
check(status == "Error", "C-level loop times out too", f"{value[:60]!r}")
check(elapsed < 8, "and also near the deadline", f"{elapsed:.1f}s")

# 4. No process is left spinning afterwards. On a machine shared with eight other
#    people, a leaked busy loop is not a cosmetic problem.
import subprocess  # noqa: E402

leaked = subprocess.run(
    ["pgrep", "-f", "-c", "generated"], capture_output=True, text=True).stdout.strip()
check(leaked in ("", "0"), "nothing left running after the kill", f"pgrep says {leaked!r}")

# 5. A missing entry point is reported, not raised.
status, value = exec_guard.run_solve("x = 1\n", timeout=5)
check(status == "Error" and "not found" in value, "missing solve() reported cleanly",
      f"{value!r}")

# 6. Assertions come back distinguishable from crashes, which is what the Test
#    operators branch on.
results = exec_guard.run_snippets([
    "assert 1 == 1",
    "assert 1 == 2, 'nope'",
    "raise ValueError('boom')",
], timeout=10)
check([r["status"] for r in results] == ["ok", "error", "error"],
      "assertion pass/fail/crash separated")
check(results[1]["error_type"] == "AssertionError"
      and results[2]["error_type"] == "ValueError",
      "error types preserved",
      f"{results[1]['error_type']}/{results[2]['error_type']}")

# 7. Snippets before a hang keep their results.
results = exec_guard.run_snippets([
    "assert 1 == 1",
    "while True:\n    pass",
    "assert 2 == 2",
], timeout=6)
check(len(results) == 3, "one result per snippet even when one hangs",
      f"{len(results)} results")
check(results[0]["status"] == "ok",
      "the snippet before the hang keeps its verdict", f"{results[0]}")
check(all(r["status"] in ("ok", "error") for r in results),
      "no snippet comes back unclassified")

# 8. Output is not truncated into a wrong verdict: a huge print does not break the
#    JSON channel.
results = exec_guard.run_snippets(["print('x' * 100000)"], timeout=10)
check(results[0]["status"] == "ok", "large stdout does not corrupt the result channel",
      f"{results[0]['status']}")

print(f"\n  {failures} failure(s)")
sys.exit(1 if failures else 0)
