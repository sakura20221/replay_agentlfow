#!/usr/bin/env python3
"""Replay the prompts that degenerated, at several presence_penalty values.

Qwen3's own model card names the cause and the remedy:

    DO NOT use greedy decoding, as it can lead to performance degradation and
    endless repetitions. ... you can adjust the presence_penalty parameter between
    0 and 2 to reduce endless repetitions. However, using a higher value may
    occasionally result in language mixing and a slight decrease in model
    performance.

So the value is a trade-off with two opposing failure modes, and picking it by
intuition is guessing. These are the actual prompts that looped, taken from the
recorded transcripts, so the measurement is on the real distribution rather than on
invented examples.

Reported per setting:
  * still_looping  -- replies whose most repeated line is >30% of the reply
  * hit_cap        -- replies that reached max_tokens (the symptom being fixed)
  * language_mix   -- CJK characters in a reply to an English prompt (the side
                      effect the card warns about)
  * median_tokens  -- whether replies get shorter for the right reason

The requests go straight to vLLM, not through the proxy: the proxy strips sampling
parameters by design, which is exactly what is under test here.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import threading
import urllib.request
from pathlib import Path

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")


def repetition_share(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return collections.Counter(lines).most_common(1)[0][1] / len(lines)


def load_looping_prompts(path: Path, limit: int) -> list[dict]:
    """Prompts whose recorded reply hit the cap and shows heavy repetition."""
    picked = []
    seen_prompts = set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (entry.get("completion_tokens") or 0) < 8000:
            continue
        if repetition_share(entry.get("completion") or "") < 0.25:
            continue
        messages = entry.get("messages") or []
        if not messages:
            continue
        key = json.dumps(messages, ensure_ascii=False)[:400]
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        picked.append({"namespace": entry.get("namespace"), "messages": messages})
        if len(picked) >= limit:
            break
    return picked


def call(url: str, messages: list[dict], penalty: float, max_tokens: int) -> dict:
    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if penalty:
        payload["presence_penalty"] = penalty
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.loads(response.read())
    choice = (body.get("choices") or [{}])[0]
    return {"text": choice.get("message", {}).get("content") or "",
            "finish": choice.get("finish_reason"),
            "tokens": (body.get("usage") or {}).get("completion_tokens", 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", default="logs/transcripts.jsonl")
    parser.add_argument("--n", type=int, default=40, help="prompts to replay per setting")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--penalties", type=float, nargs="+", default=[0, 0.5, 1.0, 1.5])
    parser.add_argument("--upstreams", nargs="+",
                        default=["http://127.0.0.1:8001/v1/chat/completions",
                                 "http://127.0.0.1:8002/v1/chat/completions"])
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    prompts = load_looping_prompts(Path(args.transcripts), args.n)
    print(f"  replaying {len(prompts)} prompt(s) that previously looped, "
          f"from {collections.Counter(p['namespace'] for p in prompts)}")
    if not prompts:
        raise SystemExit("no looping prompts found in the transcript")

    print(f"\n  {'presence_penalty':>17}{'still looping':>15}{'hit cap':>10}"
          f"{'lang mix':>10}{'median tok':>12}{'errors':>8}")
    for penalty in args.penalties:
        results: list[dict] = []
        errors = 0
        lock = threading.Lock()
        index = [0]

        def worker(slot: int) -> None:
            nonlocal errors
            while True:
                with lock:
                    if index[0] >= len(prompts):
                        return
                    i = index[0]
                    index[0] += 1
                url = args.upstreams[i % len(args.upstreams)]
                try:
                    out = call(url, prompts[i]["messages"], penalty, args.max_tokens)
                except Exception:  # noqa: BLE001
                    with lock:
                        errors += 1
                    continue
                with lock:
                    results.append(out)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        looping = sum(1 for r in results if repetition_share(r["text"]) > 0.30)
        capped = sum(1 for r in results if r["finish"] == "length")
        mixed = sum(1 for r in results if CJK.search(r["text"]))
        median = int(statistics.median([r["tokens"] for r in results])) if results else 0
        n = max(len(results), 1)
        print(f"  {penalty:>17}{looping:>8} ({looping / n:>4.0%}){capped:>5} ({capped / n:>4.0%})"
              f"{mixed:>4} ({mixed / n:>4.0%}){median:>12,}{errors:>8}")


if __name__ == "__main__":
    main()
