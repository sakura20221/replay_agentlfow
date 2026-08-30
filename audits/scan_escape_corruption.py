#!/usr/bin/env python3
"""Which prompt constants reach the model with control characters instead of LaTeX?

A triple-quoted Python string is not raw, so `\\boxed` in the source becomes the
single byte 0x08 followed by "oxed". The file still *reads* as `\\boxed{}` in an
editor and in `git show`, which is why this survives review: the corruption only
exists at runtime. MaAS's MATH template says "Present the final answer enclosed in
\\boxed{} LaTeX notation" and the model is actually shown "enclosed in <BS>oxed{}".

So every prompt module is imported, not grepped, and its string constants are
inspected for the control characters that a mis-escaped LaTeX command produces.
Grading depends on the answer format, so a garbled format instruction costs the
affected method points for a reason that has nothing to do with its workflow.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Control characters a mis-escaped LaTeX command can produce, with the command
# letter each one came from. \n and \r are excluded: those are ordinary line
# endings in a prompt and carry no LaTeX meaning here.
SUSPECT = {"\x07": "a", "\x08": "b", "\x0b": "v", "\x0c": "f", "\x09": "t"}

PATTERNS = [
    "third_party/maas/maas/ext/maas/scripts/optimized/*/*/template/*prompt*.py",
    "third_party/daao/daao/ext/maas/scripts/optimized/*/*/template/*prompt*.py",
    "third_party/aflow/workspace/*/workflows/template/*prompt*.py",
    "third_party/aflow/scripts/prompts/*.py",
    "third_party/flowbank/DiverseFlow/workspace/*/workflows/template/*prompt*.py",
    "third_party/flowbank/DiverseFlow/scripts/prompts/*.py",
    "third_party/gdesigner/GDesigner/prompt/*.py",
    "third_party/card/GDesigner/prompt/*.py",
    "third_party/masrouter/MAR/Prompts/*.py",
]


def constants(path: Path, dataset: str) -> list[tuple[str, str]]:
    """Module-level string constants, as the model would really see them.

    The module is EXECUTED, not parsed. An earlier version used
    ast.literal_eval on each assignment, which is faster and needs no imports --
    but it reports the value of the literal in the source, and the repair for this
    very corruption is applied at import time by an appended block. So the static
    reading said "92 corrupted" on files whose runtime values were already clean,
    which is precisely the class of mistake this scanner exists to catch.

    SHIM_DATASET is set for the duration because the same module yields different
    prompts per dataset by design.
    """
    previous = os.environ.get("SHIM_DATASET")
    os.environ["SHIM_DATASET"] = dataset
    try:
        namespace: dict = {"__name__": "escape_scan_probe"}
        exec(compile(path.read_text(encoding="utf-8", errors="replace"),
                     str(path), "exec"), namespace)
    except Exception:  # noqa: BLE001 - fall back to the static reading
        return _static_constants(path)
    finally:
        if previous is None:
            os.environ.pop("SHIM_DATASET", None)
        else:
            os.environ["SHIM_DATASET"] = previous
    # Leading-underscore names are excluded: those are a shim's own temporaries,
    # not text any operator sends. Anything else is fair game, including constants
    # that do not end in _PROMPT.
    return [(k, v) for k, v in namespace.items()
            if isinstance(v, str) and not k.startswith("_")]


def _static_constants(path: Path) -> list[tuple[str, str]]:
    """Used only for modules that cannot be executed standalone."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        if not isinstance(value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found.append((target.id, value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    files = [p for pattern in PATTERNS for p in sorted(ROOT.glob(pattern))
             if "__pycache__" not in p.parts]
    affected: dict[str, int] = {}
    total_constants = 0
    # Every dataset, because the same module yields different prompts per dataset:
    # a corruption that only survives on one of them still reaches the model.
    datasets = ("SHARED_MATH", "SHARED_AMC", "SHARED_MBPP", "SHARED_DROP", "SHARED_MMLUPRO")

    for path in files:
        hits: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for dataset in datasets:
            for name, text in constants(path, dataset):
                total_constants += 1
                for char, letter in SUSPECT.items():
                    if char not in text or (name, char) in seen:
                        continue
                    seen.add((name, char))
                    index = text.index(char)
                    context = text[max(0, index - 24): index + 14]
                    hits.append((f"{name} [{dataset[7:].lower()}]",
                                 f"\\{letter} -> 0x{ord(char):02x}", context))
        if not hits:
            continue
        method = path.relative_to(ROOT).parts[1]
        affected[method] = affected.get(method, 0) + len(hits)
        if args.quiet:
            continue
        print(f"\n  {path.relative_to(ROOT)}")
        for name, what, context in hits:
            printable = re.sub(r"[\x00-\x1f]", lambda m: f"<{ord(m.group()):02x}>", context)
            print(f"      {name:<28}{what:<14}...{printable}...")

    print(f"\n  scanned {len(files)} prompt module(s), {total_constants} string constant(s)")
    if affected:
        for method, count in sorted(affected.items(), key=lambda kv: -kv[1]):
            print(f"    {method:<12}{count} corrupted constant occurrence(s)")
    else:
        print("    no control characters in any prompt constant")


if __name__ == "__main__":
    main()
