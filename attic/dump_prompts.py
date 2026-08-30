#!/usr/bin/env python3
"""Print every prompt string a flagged cell sends, so mismatches can be read.

The point is to separate two things that look alike in a diff:

  task identity  -- which dataset this is, what the model is being asked to produce,
                    and any worked example. Wrong here means the model is doing the
                    wrong job, and it must be corrected.
  strategy       -- roles, reasoning steps, output format, debate structure. This is
                    the method under test and must not be touched.

So each constant is printed whole, with the phrases that identify a task marked, and
nothing is edited here.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Phrases that assert what the task is. These are the only candidates for editing.
TASK_PHRASES = [
    (r"HumanEval|GSM8K|MBPP|HotpotQA|MMLU|DROP|MATH benchmark", "names a benchmark"),
    (r"mathematical problem|math problem|solve the (?:given )?math", "says maths"),
    (r"self-contained code|python function|code block|write .{0,12}code", "says write code"),
    (r"multiple[- ]choice|option letter|\(A\)|choices", "says multiple choice"),
    (r"passage|paragraph|excerpt", "says reading"),
    (r"Demonstration Examples|Example \d|For example.{0,40}\$", "carries a worked example"),
]

TARGETS = {
    "maas/SHARED_DROP": "third_party/maas/maas/ext/maas/scripts/optimized/SHARED_DROP/train/template",
    "maas/SHARED_MMLUPRO": "third_party/maas/maas/ext/maas/scripts/optimized/SHARED_MMLUPRO/train/template",
    "maas/SHARED_MBPP": "third_party/maas/maas/ext/maas/scripts/optimized/SHARED_MBPP/train/template",
    "aflow/SHARED_MMLUPRO": "third_party/aflow/workspace/SHARED_MMLUPRO/workflows/template",
    "aflow/SHARED_DROP": "third_party/aflow/workspace/SHARED_DROP/workflows/template",
    "gdesigner/prompts": "third_party/gdesigner/GDesigner/prompt",
}


def constants(path: Path) -> list[tuple[str, str]]:
    """Module-level string constants, which is how every repo stores prompts."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.append((target.id, node.value.value))
    return found


def mark(text: str) -> list[str]:
    hits = []
    for pattern, label in TASK_PHRASES:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hits.append(f"{label}: {match.group(0)!r}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True, choices=sorted(TARGETS))
    parser.add_argument("--full", action="store_true", help="print whole prompts")
    parser.add_argument("--chars", type=int, default=500)
    args = parser.parse_args()

    base = ROOT / TARGETS[args.cell]
    if not base.exists():
        raise SystemExit(f"missing {base}")
    for path in sorted(base.glob("*.py")):
        if path.name in ("__init__.py", "operator_an.py", "operator_registry.py"):
            continue
        found = constants(path)
        if not found:
            continue
        print(f"\n{'=' * 92}\n### {path.relative_to(ROOT)}  ({len(found)} prompt constant(s))")
        for name, text in found:
            hits = mark(text)
            print(f"\n  --- {name}  [{len(text)} chars] ---")
            if hits:
                print(f"      TASK MARKERS: {'; '.join(hits)}")
            body = text if args.full else text[: args.chars]
            for line in body.splitlines():
                print("      " + line)
            if not args.full and len(text) > args.chars:
                print(f"      ... [{len(text) - args.chars} more chars]")


if __name__ == "__main__":
    main()
