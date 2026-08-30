#!/usr/bin/env python3
"""Execute the exact solutions a stalled job was about to test.

Two earlier attempts failed for the same reason: they looked for code in ```fences
and the operators pass code as plain text. This extracts by structure instead --
from the first `def` to the end of the block -- and runs each candidate in a child
process with a hard timeout.

The target is the last ScEnsemble call before the stall: its prompt carries the
candidate solutions, and the Test operator execs the winner immediately afterwards
with no timeout of its own.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from pathlib import Path


def extract_snippets(text: str) -> list[str]:
    """Code blocks, however the operator happened to format them."""
    found = []
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    found.extend(fenced)
    if not fenced:
        # Plain text: take each `def ...` and everything indented under it, plus any
        # trailing lines, which is how the operators concatenate solutions.
        for match in re.finditer(r"^(?:def |import |from )", text, re.MULTILINE):
            found.append(text[match.start():])
            break
    # A/B/C labelled solutions inside one ScEnsemble prompt.
    for part in re.split(r"\n[A-C]:\s*\n", text)[1:]:
        if "def " in part:
            found.append(part.split("\n\n\n")[0])
    return [snippet for snippet in found if "def " in snippet]


def _run(source: str) -> None:
    try:
        exec(source, {})  # noqa: S102 - this is what the Test operator does
    except Exception:  # noqa: BLE001
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="train/aflow")
    parser.add_argument("--from-time", required=True)
    parser.add_argument("--to-time", required=True)
    parser.add_argument("--timeout", type=float, default=4.0)
    args = parser.parse_args()

    entries = []
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
        if args.from_time <= stamp <= args.to_time:
            entries.append(entry)
    entries.sort(key=lambda e: e.get("timestamp") or "")
    print(f"  {len(entries)} exchange(s) in [{args.from_time}, {args.to_time}]")

    checked = hung = 0
    for entry in entries:
        texts = [entry.get("completion") or ""]
        texts += [str(m.get("content", "")) for m in (entry.get("messages") or [])]
        seen = set()
        for text in texts:
            for snippet in extract_snippets(text):
                key = snippet[:200]
                if key in seen:
                    continue
                seen.add(key)
                checked += 1
                if hangs(snippet, args.timeout):
                    hung += 1
                    loops = [line.strip() for line in snippet.splitlines()
                             if re.match(r"\s*(while|for)\b", line)]
                    print(f"\n  HANGS (> {args.timeout}s)  from the call at "
                          f"{entry.get('timestamp')}")
                    print(f"    loops: {loops[:5]}")
                    print(f"    code:\n" + "\n".join("      " + line
                                                     for line in snippet.splitlines()[:22]))
    print(f"\n  {checked} snippet(s) executed, {hung} hung")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    raise SystemExit(main())
