#!/usr/bin/env python3
"""Does re-issuing the identical request break a repetition loop?

The appeal of this over both continuation and a sampling change: the request is
byte-identical, no follow-up prompt is injected, and temperature stays at 0. If it
works, the protocol keeps its purity and the fix is a retry policy.

Why it can work at all despite greedy decoding: vLLM's continuous batching makes
the arithmetic depend on which requests share a batch, so the same prompt is not
bitwise reproducible. Measured earlier: the same question produced 429 and 427
tokens on two runs. A loop is a fragile trajectory; a different batch can miss it.

Reported per attempt: how many of the prompts are *still* looping, i.e. what a
retry-until-clean policy would have to pay for.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, "shared")
import bench  # noqa: E402

URLS = ["http://127.0.0.1:8001/v1/chat/completions",
        "http://127.0.0.1:8002/v1/chat/completions"]
FORMAT = re.compile(r"(?:Answer:\s*\(?[A-J]\)?|Answer:\s*\S+|\\boxed\{)", re.IGNORECASE)


def line_repetition(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    return collections.Counter(lines).most_common(1)[0][1] / len(lines)


def distinct_ngram_ratio(text: str, n: int = 20) -> float:
    words = text.split()
    if len(words) < n * 3:
        return 1.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def is_degenerate(text: str) -> bool:
    return line_repetition(text) >= 0.30 or distinct_ngram_ratio(text) <= 0.55


def gold_index() -> dict:
    index = {}
    for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
        rows = list(bench.load(dataset))
        search = Path("shared/data") / (dataset + "_search.jsonl")
        if search.exists():
            rows += [json.loads(line) for line in
                     search.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            key = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()[:300]
            index.setdefault(key, (dataset, row))
    return index


def load_looping(path: Path, limit: int, window: int, max_tokens: int) -> list[dict]:
    picked, seen = [], set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (entry.get("completion_tokens") or 0) < 8000:
            continue
        if not is_degenerate(entry.get("completion") or ""):
            continue
        if (entry.get("prompt_tokens") or 0) + max_tokens > window:
            continue
        messages = entry.get("messages") or []
        if not messages:
            continue
        key = json.dumps(messages, ensure_ascii=False)[:400]
        if key in seen:
            continue
        seen.add(key)
        picked.append({"messages": messages})
        if len(picked) >= limit:
            break
    return picked


def call(url, messages, max_tokens, temperature=0, penalty=0.0):
    """The sweep's request, optionally with the penalty under test.

    penalty=0 reproduces the current protocol exactly; a non-zero value tests the
    combination the retry is layered on top of.
    """
    payload = {"model": "Qwen/Qwen3-8B", "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if penalty:
        payload["presence_penalty"] = penalty
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST")
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.loads(response.read())
    choice = (body.get("choices") or [{}])[0]
    return (choice.get("message", {}).get("content") or "",
            choice.get("finish_reason"),
            (body.get("usage") or {}).get("completion_tokens", 0))


def run(items, max_tokens, concurrency, temperature=0, penalty=0.0):
    out, lock, idx, errors = {}, threading.Lock(), [0], [0]

    def worker():
        while True:
            with lock:
                if idx[0] >= len(items):
                    return
                i = idx[0]
                idx[0] += 1
            try:
                text, finish, tokens = call(URLS[i % 2], items[i]["messages"],
                                            max_tokens, temperature, penalty)
            except Exception:  # noqa: BLE001
                with lock:
                    errors[0] += 1
                continue
            with lock:
                out[i] = (text, finish, tokens)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return out, errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--window", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--penalty", type=float, default=0.0,
                        help="presence_penalty applied to every attempt")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    index = gold_index()
    prompts = load_looping(Path("logs/transcripts.jsonl"), args.n,
                           args.window, args.max_tokens)
    for entry in prompts:
        text = re.sub(r"\s+", " ",
                      str(entry["messages"][-1].get("content", ""))).strip()
        entry["gold"] = index.get(text[:300])
    print("  %d prompt(s) that looped in the sweep; %d with a gold answer"
          % (len(prompts), sum(1 for e in prompts if e.get("gold"))), flush=True)
    print("  settings: temperature=%s presence_penalty=%s max_tokens=%d"
          % (args.temperature, args.penalty, args.max_tokens), flush=True)
    print("  each attempt re-issues the identical request; nothing is appended\n",
          flush=True)

    # Items still looping after each attempt. A retry policy only re-issues these.
    pending = list(range(len(prompts)))
    best = {}
    print("  %-9s%9s%14s%14s%13s%16s" % ("attempt", "retried", "still looping",
                                         "broke free", "cumulative", "accuracy so far"),
          flush=True)
    for attempt in range(1, args.attempts + 1):
        subset = [prompts[i] for i in pending]
        results, errors = run(subset, args.max_tokens, args.concurrency,
                              args.temperature, args.penalty)
        still, freed = [], 0
        for local, global_index in enumerate(pending):
            if local not in results:
                still.append(global_index)      # an error counts as unresolved
                continue
            text, finish, _ = results[local]
            if finish == "length" and is_degenerate(text):
                still.append(global_index)
            else:
                freed += 1
                best[global_index] = text
        scores = []
        for global_index, text in best.items():
            entry = prompts[global_index]
            if not entry.get("gold"):
                continue
            dataset, row = entry["gold"]
            try:
                value, _ = bench.score(dataset, row, text)
                scores.append(float(value))
            except Exception:  # noqa: BLE001
                pass
        accuracy = ("%.3f (n=%d)" % (sum(scores) / len(scores), len(scores))
                    if scores else "n/a")
        print("  %-9d%9d%8d (%3.0f%%)%8d (%3.0f%%)%9d/%d%16s"
              % (attempt, len(subset), len(still), 100 * len(still) / max(len(subset), 1),
                 freed, 100 * freed / max(len(subset), 1),
                 len(prompts) - len(still), len(prompts), accuracy), flush=True)
        pending = still
        if not pending:
            break

    print("\n  after %d attempt(s): %d of %d prompts still loop (%.0f%%)"
          % (args.attempts, len(pending), len(prompts),
             100 * len(pending) / max(len(prompts), 1)))
    print("  a retry-until-clean policy pays one extra generation per line above,")
    print("  i.e. only for the calls that were already looping.")


if __name__ == "__main__":
    main()
