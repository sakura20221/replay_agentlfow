#!/usr/bin/env python3
"""Snapshot every prompt file before adaptation, and diff against it afterwards.

Adapting a prompt to its dataset and rewriting the author's design look identical
in a file; the difference only shows in a diff. So the baseline is taken first, and
the report at the end has to show that every changed line is task identity -- which
dataset, what the model is asked to produce, which worked example -- and that roles,
reasoning steps, output formats and debate structure are untouched.

    python prompt_baseline.py --save          # before editing
    python prompt_baseline.py --diff          # after editing
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "archive" / "prompt_baseline"

# Every file that carries prompt text for any method.
PATTERNS = [
    "third_party/maas/maas/ext/maas/scripts/optimized/*/*/template/op_prompt.py",
    "third_party/maas/maas/ext/maas/scripts/optimized/*/*/template/prompt.py",
    "third_party/daao/daao/ext/maas/scripts/optimized/*/*/template/op_prompt.py",
    "third_party/daao/daao/ext/maas/scripts/optimized/*/*/template/prompt.py",
    "third_party/aflow/workspace/*/workflows/template/op_prompt.py",
    "third_party/aflow/workspace/*/workflows/template/prompt.py",
    "third_party/aflow/scripts/prompts/*.py",
    "third_party/flowbank/DiverseFlow/workspace/*/workflows/template/*prompt*.py",
    "third_party/flowbank/DiverseFlow/scripts/prompts/*.py",
    "third_party/gdesigner/GDesigner/prompt/*.py",
    "third_party/card/GDesigner/prompt/*.py",
    "third_party/masrouter/MAR/Prompts/*.py",
]

# Lines that state what the task is. Only these may change.
TASK_WORDS = ("humaneval", "gsm8k", "mbpp", "hotpotqa", "mmlu", "drop", "math",
              "mathematical", "multiple-choice", "multiple choice", "passage",
              "code", "function", "benchmark", "demonstration examples",
              "option letter", "boxed", "frac", "times", "commonsense",
              "reading comprehension", "span")


def files() -> list[Path]:
    found: list[Path] = []
    for pattern in PATTERNS:
        found.extend(sorted(ROOT.glob(pattern)))
    return [p for p in found if "__pycache__" not in p.parts]


def save() -> None:
    if BASELINE.exists():
        print(f"  baseline already exists at {BASELINE.relative_to(ROOT)}; "
              f"refusing to overwrite it -- it is the only copy of the originals")
        return
    manifest = {}
    for path in files():
        relative = path.relative_to(ROOT)
        target = BASELINE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (BASELINE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  saved {len(manifest)} prompt file(s) to {BASELINE.relative_to(ROOT)}")


# A shim that only appends leaves the author's text byte-identical, which is a
# stronger guarantee than any line-by-line reading of a diff can give -- so those
# files are reported as one line each instead of thousands. Files with even one
# edited or deleted line get the full treatment, because that is where an
# adaptation could have quietly become a rewrite.
SHIM_MARKER = "--- shared-layer shim (agent_wf_v2) ---"


def diff() -> int:
    if not BASELINE.exists():
        raise SystemExit("no baseline; run --save first")
    manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    changed = suspicious = appended_only = 0
    for relative, digest in sorted(manifest.items()):
        current = ROOT / relative
        if not current.exists():
            print(f"\n  DELETED {relative}")
            changed += 1
            continue
        if hashlib.sha256(current.read_bytes()).hexdigest() == digest:
            continue
        changed += 1
        before = (BASELINE / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        after = current.read_text(encoding="utf-8", errors="replace").splitlines()

        # Pure append: the original is a prefix of the new file, and everything
        # after it is one shim block. Nothing the author wrote can have moved.
        if after[: len(before)] == before and SHIM_MARKER in "\n".join(after[len(before):]):
            appended_only += 1
            added = len(after) - len(before)
            print(f"  APPENDED-ONLY {relative}  (+{added} shim line(s), "
                  f"author text byte-identical)")
            continue

        print(f"\n{'=' * 92}\n  EDITED IN PLACE {relative}")
        for line in difflib.unified_diff(before, after, lineterm="", n=1,
                                         fromfile="original", tofile="adapted"):
            if line.startswith(("---", "+++", "@@")):
                print("    " + line)
                continue
            if not line.startswith(("+", "-")):
                continue
            body = line[1:].strip().lower()
            about_task = any(word in body for word in TASK_WORDS)
            marker = "     " if about_task else "  !! "
            if not about_task and body:
                suspicious += 1
            print(f"  {marker}{line}")

    print(f"\n  {changed} file(s) changed: {appended_only} append-only, "
          f"{changed - appended_only} edited in place")
    print(f"  {suspicious} edited line(s) do NOT mention any task-identity word")
    if suspicious:
        print("  Lines marked !! need justifying: they may be edits to the author's design.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", action="store_true")
    group.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    if args.save:
        save()
    else:
        raise SystemExit(diff())


if __name__ == "__main__":
    main()
