#!/usr/bin/env python3
"""Do the code snippets a stalled job last received actually hang?

The stall diagnosis so far is circumstantial: five of six stalled jobs are MBPP,
`exec_code` carries no timeout, and one thread spins at ~100% CPU. None of that
proves the spinning thread is running model-written code.

This checks it directly. The LLM calls completed -- it is the *execution* that
hangs -- so the code is in the transcript. Each snippet from just before the stall
is run in a separate process with a hard timeout. A snippet that outlives the
timeout is the mechanism, demonstrated rather than inferred.

    python test_generated_code_hangs.py --namespace train/aflow --before "2026-08-22T14:33"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
from pathlib import Path

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _run(source: str) -> None:
    """Execute in a child process so a hang can be killed."""
    namespace: dict = {}
    try:
        exec(source, namespace)  # noqa: S102 - reproducing what the operator does
    except Exception:  # noqa: BLE001 - an exception is a normal outcome here
        pass


def hangs(source: str, timeout: float) -> bool:
    process = multiprocessing.Process(target=_run, args=(source,))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="train/aflow")
    parser.add_argument("--before", required=True,
                        help="ISO prefix of the stall time, e.g. 2026-08-22T14:33")
    parser.add_argument("--tail", type=int, default=40,
                        help="how many of the last exchanges before the stall to check")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    candidates = []
    for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("namespace") != args.namespace:
            continue
        timestamp = entry.get("timestamp") or ""
        if timestamp > args.before:
            continue
        candidates.append(entry)
    candidates = candidates[-args.tail:]
    print(f"  {len(candidates)} exchange(s) from {args.namespace} up to {args.before}")

    checked = hung = 0
    for entry in candidates:
        # Code appears in the *prompt* as well as the reply: a ScEnsemble call sends
        # the candidate solutions (which are code) and gets back only a letter. An
        # earlier version of this test only read the reply and therefore found
        # nothing in exactly the window where the stall happened.
        haystacks = [entry.get("completion") or ""]
        haystacks += [str(m.get("content", "")) for m in (entry.get("messages") or [])]
        for block in [b for text in haystacks for b in CODE_BLOCK.findall(text)]:
            if "def " not in block and "for " not in block and "while " not in block:
                continue
            checked += 1
            if hangs(block, args.timeout):
                hung += 1
                # Show enough to identify the loop, not the whole snippet.
                loops = [line.strip() for line in block.splitlines()
                         if re.match(r"\s*(while|for)\b", line)]
                print(f"\n  HANGS (> {args.timeout}s) at {entry.get('timestamp')}")
                print(f"    loop lines: {loops[:4]}")
                print(f"    first 200 chars: {block[:200]!r}")
    print(f"\n  {checked} snippet(s) executed, {hung} outlived the {args.timeout}s timeout")
    if checked == 0:
        print("  (no executable snippet found -- the stall is not explained by this)")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    raise SystemExit(main())
