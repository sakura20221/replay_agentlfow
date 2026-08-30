#!/usr/bin/env python3
"""Settle presence_penalty at the real output budget, with the causes separated.

The earlier probe ran at max_tokens=2048, which conflates two populations: a reply
that loops forever and a reply that simply needed 3,000 tokens both hit the cap. At
the real 8,192 budget that confusion mostly disappears -- measured over the whole
sweep, 93% of capped replies are degenerate -- so this run reports the split rather
than one undifferentiated rate:

    capped + looping   what the penalty is meant to remove
    capped + long      a genuinely long reply, which no penalty should touch
    finished           reached a natural end

n=200 because at n=30 the 95% interval on the reduction factor spanned 0.12x-1.23x,
i.e. "no effect at all" was not excluded. At n=200 it narrows to roughly ten points,
enough to decide whether to discard 24 finished jobs and start over.

Requests go straight to vLLM, not through the proxy: the proxy strips sampling
parameters by design, which is the thing under test.
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

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
FORMAT = re.compile(r"(?:Answer:\s*\(?[A-J]\)?|Answer:\s*\S+|\\boxed\{)", re.IGNORECASE)
URLS = ["http://127.0.0.1:8001/v1/chat/completions",
        "http://127.0.0.1:8002/v1/chat/completions"]


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
    """Either a verbatim loop or a near-loop that varies slightly each cycle."""
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
    """Recorded prompts that looped, restricted to those that still fit the window.

    The fit check matters: a request that would 400 comes back as an empty reply,
    and an error counted as "produced no answer" would read as a quality result.
    """
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


def call(url, messages, temperature, penalty, max_tokens):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--window", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()

    index = gold_index()
    prompts = load_looping(Path("logs/transcripts.jsonl"), args.n,
                           args.window, args.max_tokens)
    for entry in prompts:
        text = re.sub(r"\s+", " ",
                      str(entry["messages"][-1].get("content", ""))).strip()
        entry["gold"] = index.get(text[:300])
    with_gold = sum(1 for entry in prompts if entry.get("gold"))
    print("  %d looping prompt(s) that fit at max_tokens=%d; %d with a gold answer"
          % (len(prompts), args.max_tokens, with_gold), flush=True)

    header = ("temp", "penalty", "capped+loop", "capped+long", "finished",
              "has answer", "CJK", "median tok", "accuracy", "err")
    print("\n  %5s%9s%14s%14s%12s%13s%6s%12s%16s%5s" % header, flush=True)

    for temperature, penalty in ((0, 0), (0, 1.0), (0.2, 1.0)):
        out, lock, index_counter, errors = [], threading.Lock(), [0], [0]

        def worker():
            while True:
                with lock:
                    if index_counter[0] >= len(prompts):
                        return
                    i = index_counter[0]
                    index_counter[0] += 1
                try:
                    text, finish, tokens = call(URLS[i % 2], prompts[i]["messages"],
                                                temperature, penalty, args.max_tokens)
                except Exception:  # noqa: BLE001 - an error is a data point, not a stop
                    with lock:
                        errors[0] += 1
                    continue
                with lock:
                    out.append((prompts[i], text, finish, tokens))

        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        n = max(len(out), 1)
        capped_loop = sum(1 for _, t, f, _ in out if f == "length" and is_degenerate(t))
        capped_long = sum(1 for _, t, f, _ in out if f == "length" and not is_degenerate(t))
        finished = sum(1 for _, _, f, _ in out if f == "stop")
        formatted = sum(1 for _, t, _, _ in out if FORMAT.search(t or ""))
        cjk = sum(1 for _, t, _, _ in out if CJK.search(t or ""))
        median = int(statistics.median([tk for _, _, _, tk in out])) if out else 0
        scores = []
        for entry, text, _, _ in out:
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
        print("  %5s%9s%8d (%3.0f%%)%8d (%3.0f%%)%6d (%3.0f%%)%7d (%3.0f%%)%6d%12s%16s%5d"
              % (temperature, penalty,
                 capped_loop, 100 * capped_loop / n,
                 capped_long, 100 * capped_long / n,
                 finished, 100 * finished / n,
                 formatted, 100 * formatted / n,
                 cjk, format(median, ","), accuracy, errors[0]), flush=True)


if __name__ == "__main__":
    main()
