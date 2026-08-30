#!/usr/bin/env python3
"""What was a job actually doing in the minutes before it stopped producing output?

The generated-code hypothesis failed its test -- no executable snippet in the last
40 exchanges -- so this looks at the exchanges themselves rather than at a theory
about them: what was sent, what came back, and what the very last call was.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="train/aflow")
    parser.add_argument("--window", default="2026-08-22T14:3",
                        help="ISO prefix selecting the minutes of interest")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    hits = []
    for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("namespace") != args.namespace:
            continue
        if not (entry.get("timestamp") or "").startswith(args.window):
            continue
        hits.append(entry)

    hits.sort(key=lambda e: e.get("timestamp") or "")
    print(f"  {len(hits)} exchange(s) in {args.namespace} matching {args.window}")
    if not hits:
        return

    print(f"  first {hits[0]['timestamp']}   last {hits[-1]['timestamp']}")

    # What kind of call were these? The operator or optimiser is identifiable from
    # recognisable fragments of the prompts each one uses.
    kinds = collections.Counter()
    for entry in hits:
        text = " ".join(str(m.get("content", ""))
                        for m in (entry.get("messages") or []))
        if "solution_letter" in text or "most consistent solution" in text:
            kinds["ScEnsemble"] += 1
        elif "prompt_custom" in text or "modify the Graph" in text or "<graph>" in text:
            kinds["optimiser (graph generation)"] += 1
        elif "Your code must pass these tests" in text:
            kinds["MBPP solve"] += 1
        elif "Passage:" in text:
            kinds["DROP solve"] += 1
        elif re.search(r"\(A\).*\(J\)", text, re.DOTALL):
            kinds["MMLU-Pro solve"] += 1
        else:
            kinds["other"] += 1
    print(f"  call kinds: {dict(kinds)}")

    print(f"\n  the last {args.show} exchange(s) before output stopped:")
    for entry in hits[-args.show:]:
        sent = str((entry.get("messages") or [{}])[-1].get("content", ""))
        got = str(entry.get("completion") or "")
        print(f"\n  --- {entry.get('timestamp')}  "
              f"tokens={entry.get('prompt_tokens')}/{entry.get('completion_tokens')} "
              f"finish={entry.get('finish_reason')} ---")
        print(f"    SENT tail: {sent[-220:]!r}")
        print(f"    GOT  head: {got[:220]!r}")
        print(f"    GOT  tail: {got[-160:]!r}")


if __name__ == "__main__":
    main()
