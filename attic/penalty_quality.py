#!/usr/bin/env python3
"""Does the penalty produce usable answers, or just shorter unusable ones?

A falling hit-cap rate is necessary but not sufficient: a reply can stop early and
still say nothing. What the experiment needs is a reply that carries an answer in
the requested shape, so that is what gets counted here -- and, where the prompt can
be matched back to a dataset item, whether that answer is actually right.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, "shared")
import bench  # noqa: E402

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
# The shapes the shared answer-format instruction asks for.
FORMAT = re.compile(r"(?:Answer:\s*\(?[A-J]\)?|Answer:\s*\S+|\\boxed\{)", re.IGNORECASE)


def build_index() -> dict[str, tuple[str, dict]]:
    """First 300 characters of question_text -> (dataset, row), for gold lookup."""
    index = {}
    for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
        for row in bench.load(dataset):
            key = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()[:300]
            index.setdefault(key, (dataset, row))
        for name in (f"{dataset}_search",):
            path = Path("shared/data") / f"{name}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = re.sub(r"\s+", " ",
                                 bench.question_text(dataset, row)).strip()[:300]
                    index.setdefault(key, (dataset, row))
    return index


def load_prompts(path: Path, limit: int) -> list[dict]:
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
        messages = entry.get("messages") or []
        if not messages:
            continue
        key = json.dumps(messages, ensure_ascii=False)[:400]
        if key in seen:
            continue
        seen.add(key)
        picked.append(entry)
        if len(picked) >= limit:
            break
    return picked


def call(url, messages, penalty, max_tokens):
    payload = {"model": "Qwen/Qwen3-8B", "messages": messages, "temperature": 0,
               "max_tokens": max_tokens,
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
            choice.get("finish_reason"),
            (body.get("usage") or {}).get("completion_tokens", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--penalties", type=float, nargs="+", default=[0, 1.0, 1.5])
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    index = build_index()
    prompts = load_prompts(Path("logs/transcripts.jsonl"), args.n)
    matched = 0
    for entry in prompts:
        text = re.sub(r"\s+", " ", str(entry["messages"][-1].get("content", ""))).strip()
        for length in (300, 250, 200):
            hit = index.get(text[:length]) if length == 300 else None
            if hit:
                entry["gold"] = hit
                matched += 1
                break
    print(f"  {len(prompts)} prompt(s); {matched} matched back to a dataset item "
          f"(gold available for those)")

    urls = ["http://127.0.0.1:8001/v1/chat/completions",
            "http://127.0.0.1:8002/v1/chat/completions"]
    print(f"\n  {'penalty':>8}{'has answer format':>20}{'finished':>11}"
          f"{'lang mix':>10}{'scored right':>15}")
    for penalty in args.penalties:
        out, lock, idx = [], threading.Lock(), [0]

        def worker():
            while True:
                with lock:
                    if idx[0] >= len(prompts):
                        return
                    i = idx[0]; idx[0] += 1
                try:
                    text, finish, tokens = call(urls[i % 2], prompts[i]["messages"],
                                                penalty, args.max_tokens)
                except Exception:  # noqa: BLE001
                    return
                with lock:
                    out.append((prompts[i], text, finish, tokens))

        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for t in threads: t.start()
        for t in threads: t.join()

        n = max(len(out), 1)
        formatted = sum(1 for _, t, _, _ in out if FORMAT.search(t or ""))
        finished = sum(1 for _, _, f, _ in out if f == "stop")
        mixed = sum(1 for _, t, _, _ in out if CJK.search(t or ""))
        graded = [(e, t) for e, t, _, _ in out if e.get("gold")]
        right = 0
        for entry, text in graded:
            dataset, row = entry["gold"]
            try:
                value, _ = bench.score(dataset, row, text)
                right += value >= 0.5
            except Exception:  # noqa: BLE001
                pass
        share = f"{right}/{len(graded)}" if graded else "n/a"
        print(f"  {penalty:>8}{formatted:>13} ({formatted / n:>4.0%}){finished:>6} "
              f"({finished / n:>4.0%}){mixed:>4} ({mixed / n:>4.0%}){share:>15}")


if __name__ == "__main__":
    main()
