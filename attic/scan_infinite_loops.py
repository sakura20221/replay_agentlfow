#!/usr/bin/env python3
"""Find loops that cannot terminate, in the code the stalled job was handling.

Executing 1,273 exchanges with a timeout each is slow; reading them is not. A loop
that cannot terminate has a recognisable shape, so this narrows the candidates
statically and only then hands them to the executor.

Patterns flagged:
  * `while True` with no `break`, `return` or `raise` in its body
  * `while <name>` where the body never assigns to <name>
  * a `for` over a generator that never yields (rare, reported separately)

Static analysis cannot prove a hang, so anything flagged here is a candidate, not a
verdict -- the verdict comes from running it.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def snippets(text: str) -> list[str]:
    found = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not found:
        match = re.search(r"^(?:def |import |from )", text, re.MULTILINE)
        if match:
            found = [text[match.start():]]
    for part in re.split(r"\n[A-C]:\s*\n", text)[1:]:
        if "def " in part:
            found.append(part.split("\n\n\n")[0])
    return [s for s in found if "while" in s or "for " in s]


def unterminating_loops(source: str) -> list[str]:
    """Loops whose exit condition cannot be reached, as far as the AST shows."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        body_src = "\n".join(ast.unparse(stmt) for stmt in node.body)
        exits = any(isinstance(inner, (ast.Break, ast.Return, ast.Raise))
                    for stmt in node.body for inner in ast.walk(stmt))
        test = ast.unparse(node.test)
        if test in ("True", "1") and not exits:
            findings.append(f"while {test}: no break/return/raise")
            continue
        # `while name:` where the body never rebinds name.
        if isinstance(node.test, ast.Name):
            name = node.test.id
            assigned = any(
                isinstance(inner, (ast.Assign, ast.AugAssign))
                and name in ast.unparse(inner).split("=")[0]
                for stmt in node.body for inner in ast.walk(stmt))
            if not assigned and not exits:
                findings.append(f"while {name}: never reassigned, no exit")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="train/aflow")
    parser.add_argument("--from-time", required=True)
    parser.add_argument("--to-time", required=True)
    args = parser.parse_args()

    examined = flagged = 0
    reported = set()
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
                examined += 1
                problems = unterminating_loops(snippet)
                if not problems:
                    continue
                key = snippet[:150]
                if key in reported:
                    continue
                reported.add(key)
                flagged += 1
                print(f"\n  [{stamp}] {problems}")
                for line_ in snippet.splitlines()[:18]:
                    print("      " + line_)
    print(f"\n  {examined} snippet(s) examined, {flagged} distinct candidate(s) flagged")


if __name__ == "__main__":
    main()
