#!/usr/bin/env python3
"""Read what a method actually sent and received. Use this before guessing.

Twice in one day a diagnosis went wrong from reasoning about code instead of
reading the recorded exchange:

* AFlow's generated rounds all scored exactly 0.0000. Two hypotheses about my own
  ScEnsemble patch were wrong; the per-item `prediction` field said
  `has no attribute 'XXX_PROMPT'` -- the optimiser had copied a placeholder.
* MaAS discarded 1020 MBPP samples. I completed `mbpp_public_test.jsonl`, which
  changed nothing, because the full error string -- not the 70 characters I had
  grepped -- named `humaneval_public_test.jsonl`.

Both were one query away in data already on disk. So the query is a tool.

    python inspect_io.py --namespace train/maas --last 3
    python inspect_io.py --grep "XXX_PROMPT" --last 5
    python inspect_io.py --namespace train/masrouter --errors
    python inspect_io.py --longest 5            # what is eating the wall clock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "logs" / "transcripts.jsonl"
CALLS = ROOT / "logs" / "api_calls.jsonl"


def read(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def show_exchange(entry: dict, width: int) -> None:
    print("=" * 100)
    print(f"  {entry.get('timestamp', '?')}  ns={entry.get('namespace')}  "
          f"tokens={entry.get('prompt_tokens')}/{entry.get('completion_tokens')}  "
          f"finish={entry.get('finish_reason')}  id={entry.get('request_id', '')[:8]}")
    for message in entry.get("messages") or []:
        content = str(message.get("content", ""))
        role = message.get("role", "?")
        if len(content) > width * 2:
            content = f"{content[:width]}\n      ... [{len(content) - width * 2} chars omitted] ...\n{content[-width:]}"
        print(f"  --- SENT ({role}) ---")
        print("      " + content.replace("\n", "\n      "))
    completion = str(entry.get("completion", ""))
    if len(completion) > width * 2:
        completion = (f"{completion[:width]}\n      ... [{len(completion) - width * 2} chars omitted] ..."
                      f"\n{completion[-width:]}")
    print("  --- RECEIVED ---")
    print("      " + completion.replace("\n", "\n      "))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", help="e.g. train/maas")
    parser.add_argument("--grep", help="show exchanges whose prompt or reply contains this")
    parser.add_argument("--last", type=int, default=2)
    parser.add_argument("--width", type=int, default=700, help="chars kept from each end")
    parser.add_argument("--errors", action="store_true",
                        help="full error strings from api_calls.jsonl, never truncated")
    parser.add_argument("--longest", type=int, metavar="N",
                        help="the N slowest completed calls")
    args = parser.parse_args()

    if args.errors:
        calls = [c for c in read(CALLS) if c.get("error") or c.get("status") == 502]
        if args.namespace:
            calls = [c for c in calls if c.get("namespace") == args.namespace]
        counts: dict[str, int] = {}
        for call in calls:
            counts[str(call.get("error"))] = counts.get(str(call.get("error")), 0) + 1
        print(f"  {len(calls)} failed call(s), {len(counts)} distinct message(s)\n")
        for message, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            # Whole string. Truncating this is exactly how the MBPP diagnosis went
            # to the wrong file for an hour.
            print(f"  [{count}x] {message}\n")
        return

    if args.longest:
        calls = [c for c in read(CALLS) if c.get("latency_ms")]
        if args.namespace:
            calls = [c for c in calls if c.get("namespace") == args.namespace]
        calls.sort(key=lambda c: -c["latency_ms"])
        for call in calls[: args.longest]:
            print(f"  {call['latency_ms'] / 1000:>8.1f}s  {call.get('namespace'):<20} "
                  f"in={call.get('prompt_tokens')} out={call.get('completion_tokens')} "
                  f"truncated={call.get('truncated')} id={call.get('request_id', '')[:8]}")
        return

    entries = read(TRANSCRIPTS)
    if args.namespace:
        entries = [e for e in entries if e.get("namespace") == args.namespace]
    if args.grep:
        needle = args.grep
        entries = [e for e in entries
                   if needle in str(e.get("completion", ""))
                   or needle in json.dumps(e.get("messages") or [], ensure_ascii=False)]
    print(f"  {len(entries)} matching exchange(s); showing the last {min(args.last, len(entries))}\n")
    for entry in entries[-args.last:]:
        show_exchange(entry, args.width)


if __name__ == "__main__":
    main()
