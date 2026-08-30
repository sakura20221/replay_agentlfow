#!/usr/bin/env python3
"""Reproduce what the Test operator really does: define the function *and call it*.

The previous attempt executed only the definitions and found no hang -- of course:
`def f(): while True: pass` defines instantly. The hang happens on invocation, and
the Test operator invokes via the MBPP asserts:

    exec(solution + "\\n" + "assert candidate(...) == ...")

So this pairs each recorded snippet with that problem's real test cases, binds the
function to `candidate` exactly as the operator does, and runs the whole thing in a
child process with a hard timeout.
"""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import re
from pathlib import Path


def defined_functions(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


def snippets(text: str) -> list[str]:
    found = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not found:
        match = re.search(r"^(?:def |import |from )", text, re.MULTILINE)
        if match:
            found = [text[match.start():]]
    for part in re.split(r"\n[A-C]:\s*\n", text)[1:]:
        if "def " in part:
            found.append(part.split("\n\n\n")[0])
    return [s for s in found if "def " in s]


def _run(source: str) -> None:
    try:
        exec(source, {})  # noqa: S102 - this is the operator's own mechanism
    except Exception:  # noqa: BLE001 - a failing assert is a normal outcome
        pass


def hangs(source: str, timeout: float) -> bool:
    process = multiprocessing.Process(target=_run, args=(source,))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(3)
        if process.is_alive():
            process.kill()
        return True
    return False


def load_tests() -> dict[str, list[str]]:
    tests: dict[str, list[str]] = {}
    for name in ("mbpp.jsonl", "mbpp_search.jsonl"):
        path = Path("shared/data", name)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entry = row.get("entry_point")
            if entry and row.get("test_list"):
                tests.setdefault(entry, row["test_list"])
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="train/aflow")
    parser.add_argument("--from-time", required=True)
    parser.add_argument("--to-time", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    tests = load_tests()
    print(f"  {len(tests)} MBPP entry point(s) with test cases", flush=True)

    seen: set[str] = set()
    checked = hung = no_tests = 0
    for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("namespace") != args.namespace:
            continue
        stamp = entry.get("timestamp") or ""
        if not (args.from_time <= stamp <= args.to_time):
            continue
        texts = [entry.get("completion") or ""]
        texts += [str(m.get("content", "")) for m in (entry.get("messages") or [])]
        for text in texts:
            for snippet in snippets(text):
                key = snippet[:180]
                if key in seen:
                    continue
                seen.add(key)
                names = defined_functions(snippet)
                cases = next((tests[n] for n in names if n in tests), None)
                if not cases:
                    no_tests += 1
                    continue
                target = next(n for n in names if n in tests)
                # Bind to `candidate`, as the operator's test harness does.
                harness = (snippet + "\n\n"
                           + f"candidate = {target}\n"
                           + "\n".join(re.sub(rf"\b{re.escape(target)}\b", "candidate", c)
                                       for c in cases))
                checked += 1
                if hangs(harness, args.timeout):
                    hung += 1
                    print(f"\n  HANGS (> {args.timeout}s)  call at {stamp}  function {target}",
                          flush=True)
                    print("    " + "\n    ".join(snippet.splitlines()[:16]), flush=True)
                    print(f"    tests: {cases[:2]}", flush=True)
    print(f"\n  {checked} snippet(s) executed with their asserts, {hung} hung; "
          f"{no_tests} had no matching test case", flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    raise SystemExit(main())
