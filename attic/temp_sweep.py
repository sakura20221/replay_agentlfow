#!/usr/bin/env python3
"""Does a little temperature help the loops, and what does it cost in variance?

Two measurements, because raising temperature trades one problem for another:

  A. On the prompts that actually looped: does (temperature, presence_penalty)
     break the loop *and* leave a usable answer?
  B. On ordinary prompts: how often does the same prompt score differently when
     run twice? That is the price of leaving greedy decoding.

Measurement B needs a baseline: temperature 0 is already not reproducible here,
because vLLM's continuous batching makes the arithmetic depend on which requests
share a batch. So B reports the flip rate at temperature 0 as well -- the question
is how much *extra* noise a temperature adds, not whether noise appears.
"""
from __future__ import annotations

import argparse
import json
import re
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
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.loads(response.read())
    choice = (body.get("choices") or [{}])[0]
    return (choice.get("message", {}).get("content") or "",
            choice.get("finish_reason"))


def run_batch(items, temperature, penalty, max_tokens, concurrency):
    out, lock, idx = [], threading.Lock(), [0]

    def worker():
        while True:
            with lock:
                if idx[0] >= len(items):
                    return
                i = idx[0]; idx[0] += 1
            try:
                text, finish = call(URLS[i % 2], items[i]["messages"],
                                    temperature, penalty, max_tokens)
            except Exception:  # noqa: BLE001
                text, finish = "", "error"
            with lock:
                out.append((items[i], text, finish))

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    return out


def gold_index():
    index = {}
    for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
        rows = list(bench.load(dataset))
        search = Path("shared/data") / f"{dataset}_search.jsonl"
        if search.exists():
            rows += [json.loads(l) for l in search.read_text(encoding="utf-8").splitlines() if l.strip()]
        for row in rows:
            key = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()[:300]
            index.setdefault(key, (dataset, row))
    return index


def attach_gold(entries, index):
    for entry in entries:
        text = re.sub(r"\s+", " ", str(entry["messages"][-1].get("content", ""))).strip()
        entry["gold"] = index.get(text[:300])
    return entries


def load(path: Path, want_loops: bool, limit: int, index) -> list[dict]:
    picked, seen = [], set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        tokens = entry.get("completion_tokens") or 0
        looped = tokens >= 8000
        if looped != want_loops:
            continue
        messages = entry.get("messages") or []
        if not messages:
            continue
        key = json.dumps(messages, ensure_ascii=False)[:400]
        if key in seen:
            continue
        text = re.sub(r"\s+", " ", str(messages[-1].get("content", ""))).strip()
        if not want_loops and text[:300] not in index:
            continue          # ordinary prompts are only useful with a gold answer
        seen.add(key)
        picked.append({"messages": messages})
        if len(picked) >= limit:
            break
    return picked


def score_of(entry, text) -> float | None:
    if not entry.get("gold"):
        return None
    dataset, row = entry["gold"]
    try:
        value, _ = bench.score(dataset, row, text)
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument("--ordinary", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    index = gold_index()
    transcripts = Path("logs/transcripts.jsonl")
    loops = attach_gold(load(transcripts, True, args.loops, index), index)
    ordinary = attach_gold(load(transcripts, False, args.ordinary, index), index)
    print(f"  A: {len(loops)} looping prompt(s), {sum(1 for e in loops if e['gold'])} with gold")
    print(f"  B: {len(ordinary)} ordinary prompt(s), all with gold")

    print(f"\n  === A. on the prompts that looped ===")
    print(f"  {'temp':>6}{'penalty':>9}{'still capped':>14}{'has answer':>13}"
          f"{'lang mix':>10}{'accuracy':>11}")
    for temperature, penalty in ((0, 1.0), (0.2, 1.0), (0.3, 1.0), (0.6, 1.0), (0.2, 0)):
        out = run_batch(loops, temperature, penalty, args.max_tokens, args.concurrency)
        n = max(len(out), 1)
        capped = sum(1 for _, _, f in out if f == "length")
        formatted = sum(1 for _, t, _ in out if FORMAT.search(t or ""))
        mixed = sum(1 for _, t, _ in out if CJK.search(t or ""))
        scores = [s for e, t, _ in out if (s := score_of(e, t)) is not None]
        acc = f"{sum(scores) / len(scores):.2f} (n={len(scores)})" if scores else "n/a"
        print(f"  {temperature:>6}{penalty:>9}{capped:>7} ({capped / n:>4.0%})"
              f"{formatted:>7} ({formatted / n:>4.0%}){mixed:>4} ({mixed / n:>4.0%}){acc:>11}")

    print(f"\n  === B. run-to-run variation on ordinary prompts ===")
    print(f"  {'temp':>6}{'penalty':>9}{'run 1 acc':>11}{'run 2 acc':>11}"
          f"{'items that flipped':>21}")
    for temperature, penalty in ((0, 0), (0, 1.0), (0.2, 1.0), (0.6, 1.0)):
        first = {id(e): s for e, t, _ in run_batch(ordinary, temperature, penalty,
                                                   args.max_tokens, args.concurrency)
                 if (s := score_of(e, t)) is not None}
        second = {id(e): s for e, t, _ in run_batch(ordinary, temperature, penalty,
                                                    args.max_tokens, args.concurrency)
                  if (s := score_of(e, t)) is not None}
        shared = set(first) & set(second)
        if not shared:
            print(f"  {temperature:>6}{penalty:>9}        n/a")
            continue
        a = sum(first[k] for k in shared) / len(shared)
        b = sum(second[k] for k in shared) / len(shared)
        flipped = sum(1 for k in shared if abs(first[k] - second[k]) > 0.4)
        print(f"  {temperature:>6}{penalty:>9}{a:>11.3f}{b:>11.3f}"
              f"{flipped:>13}/{len(shared)} ({flipped / len(shared):>4.0%})")


if __name__ == "__main__":
    main()
