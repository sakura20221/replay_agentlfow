#!/usr/bin/env python3
"""Re-send the exact prompts that overflowed the context window.

The prompts are taken from the recorded transcripts by request_id, so this is the
same text the method built, not a reconstruction. It answers two questions that
arithmetic alone cannot: whether the wider window actually serves them, and how
much of it they use once served -- the second matters because a method that
accumulates other agents' output may simply grow into whatever room it is given.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY = "http://127.0.0.1:18080/replay/overflow/v1/chat/completions"


def overflow_request_ids(log: Path) -> set[str]:
    ids = set()
    with log.open(errors="replace") as handle:
        for line in handle:
            if "maximum context length" not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("request_id"):
                ids.add(record["request_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="logs/api_calls.jsonl")
    parser.add_argument("--transcripts", default="logs/transcripts.jsonl")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    wanted = overflow_request_ids(ROOT / args.log)
    if not wanted:
        raise SystemExit("no context-overflow records in the log")

    messages_by_id: dict[str, list] = {}
    with (ROOT / args.transcripts).open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("{"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("request_id") in wanted and entry.get("messages"):
                messages_by_id[entry["request_id"]] = entry["messages"]

    print(f"  {len(wanted)} overflowing request(s) recorded; "
          f"{len(messages_by_id)} had their prompt captured\n")
    if not messages_by_id:
        print("  The transcript does not carry them: the proxy writes a transcript line")
        print("  only for SUCCESSFUL calls, and these never succeeded. Replaying needs")
        print("  the prompt, so this can only be measured on a fresh run.")
        return

    served = 0
    prompt_tokens = []
    for request_id, messages in messages_by_id.items():
        body = {"model": "Qwen/Qwen3-8B", "messages": messages,
                "max_tokens": args.max_tokens}
        request = urllib.request.Request(
            PROXY, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"    {request_id[:8]}  FAILED  {type(exc).__name__}: {str(exc)[:90]}")
            continue
        usage = parsed.get("usage") or {}
        used = usage.get("prompt_tokens", 0)
        content = ((parsed.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if used:
            prompt_tokens.append(used)
        ok = bool(str(content).strip())
        served += 1 if ok else 0
        print(f"    {request_id[:8]}  {'served' if ok else 'EMPTY '}  "
              f"prompt {used:,} tok, reply {usage.get('completion_tokens', 0):,} tok")

    print(f"\n  {served}/{len(messages_by_id)} now answered with real content")
    if prompt_tokens:
        print(f"  prompt sizes: min {min(prompt_tokens):,}  median "
              f"{statistics.median(prompt_tokens):,.0f}  max {max(prompt_tokens):,}")
        print(f"  headroom left in a 40,960 window: "
              f"{40960 - max(prompt_tokens):,} tokens")


if __name__ == "__main__":
    main()
