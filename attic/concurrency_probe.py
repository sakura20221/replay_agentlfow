#!/usr/bin/env python3
"""Where does throughput stop rising as client concurrency goes up?

`--jobs` was set to 12 by guesswork. Raising it only helps if the serving side has
headroom; if it does not, more jobs just deepen a queue and every job gets slower,
which looks the same from outside as progress. So the ceiling is measured rather
than assumed, on an idle machine, through the proxy -- the proxy is a
ThreadingHTTPServer doing JSON work per call in Python, so it is itself a candidate
bottleneck and has to be inside the measurement, not beside it.

Reported per level: completed requests/sec, mean and p95 latency, and vLLM's own
Running/Waiting counts. Waiting > 0 with flat throughput is the ceiling.

Caveat recorded in the output: the GPUs are shared with other users, so absolute
throughput is a lower bound and the *shape* of the curve is what to read.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import json
import re
import statistics
import time
import urllib.error
import urllib.request

PROXY = "http://127.0.0.1:18080/probe/concurrency/v1/chat/completions"
METRIC_PORTS = (8001, 8002)
TRANSCRIPTS = "logs/transcripts.jsonl"

# The load is REPLAYED from what the methods actually sent, not synthesised.
#
# A first version asked "what is 17 * 23?" behind 1,200 tokens of filler. The
# model answered in about four tokens, so 105,311 requests went through in four
# minutes and the measurement was almost pure prefill -- nothing like a workflow
# whose replies run to hundreds of tokens of reasoning. Since decode is what the
# real jobs spend their time on, sizing concurrency off that number would have
# been sizing it off the wrong bottleneck.
#
# Replaying recorded prompts fixes both halves at once: prompt length and reply
# length come out of the real distribution, because they are the real prompts.


def load_prompts(path: str, wanted: int, seed: int) -> list[list[dict]]:
    """Reservoir-sample recorded message lists, so no one method dominates.

    Sampling rather than taking the head: the file is written in job order, so its
    first thousand lines are one method on one dataset.
    """
    import random

    rng = random.Random(seed)
    reservoir: list[list[dict]] = []
    seen = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                messages = json.loads(line).get("messages")
            except json.JSONDecodeError:
                continue
            if not messages:
                continue
            seen += 1
            if len(reservoir) < wanted:
                reservoir.append(messages)
            else:
                index = rng.randrange(seen)
                if index < wanted:
                    reservoir[index] = messages
    return reservoir


def one_call(messages: list[dict], max_tokens: int, timeout: float) -> tuple[bool, float, int]:
    body = {"model": "Qwen/Qwen3-8B", "messages": messages, "max_tokens": max_tokens}
    started = time.monotonic()
    request = urllib.request.Request(
        PROXY, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        tokens = (parsed.get("usage") or {}).get("completion_tokens", 0)
        return True, time.monotonic() - started, tokens
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, ConnectionError):
        return False, time.monotonic() - started, 0


_RUNNING = re.compile(r"vllm:num_requests_running\S*\s+([0-9.]+)")
_WAITING = re.compile(r"vllm:num_requests_waiting\S*\s+([0-9.]+)")
_KV = re.compile(r"vllm:(?:gpu_)?(?:cache_usage_perc|kv_cache_usage_perc)\S*\s+([0-9.]+)")


def vllm_state() -> str:
    """Running/Waiting/KV per instance, from the Prometheus endpoint."""
    parts = []
    for port in METRIC_PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
                text = response.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - a missing metrics port must not stop the probe
            parts.append(f"{port}:?")
            continue

        def total(pattern: re.Pattern) -> float:
            return sum(float(v) for v in pattern.findall(text))

        parts.append(f"{port}: run={total(_RUNNING):.0f} wait={total(_WAITING):.0f} "
                     f"kv={total(_KV) * 100:.0f}%")
    return "  |  ".join(parts)


def level(concurrency: int, seconds: float, max_tokens: int, timeout: float,
          prompts: list[list[dict]]) -> dict:
    """Keep `concurrency` requests in flight for `seconds`, then report."""
    latencies: list[float] = []
    tokens = 0
    failures = 0
    deadline = time.monotonic() + seconds
    mid_state = ""
    started = time.monotonic()
    cursor = itertools.cycle(prompts)

    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {pool.submit(one_call, next(cursor), max_tokens, timeout)
                   for _ in range(concurrency)}
        while pending:
            done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in done:
                ok, elapsed, completion_tokens = future.result()
                if ok:
                    latencies.append(elapsed)
                    tokens += completion_tokens
                else:
                    failures += 1
                if time.monotonic() < deadline:
                    pending.add(pool.submit(one_call, next(cursor), max_tokens, timeout))
            if not mid_state and time.monotonic() - started > seconds * 0.5:
                mid_state = vllm_state()

    wall = time.monotonic() - started
    return {
        "concurrency": concurrency,
        "completed": len(latencies),
        "failures": failures,
        "wall_s": wall,
        "rps": len(latencies) / wall if wall else 0.0,
        "tok_per_s": tokens / wall if wall else 0.0,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_s": (statistics.quantiles(latencies, n=20)[-1]
                          if len(latencies) >= 20 else max(latencies, default=0.0)),
        "vllm": mid_state or vllm_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, nargs="+", default=[16, 32, 64, 128, 192])
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="the real protocol's per-reply cap")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--transcripts", default=TRANSCRIPTS)
    args = parser.parse_args()

    prompts = load_prompts(args.transcripts, args.samples, seed=20260823)
    if not prompts:
        raise SystemExit(f"no recorded prompts in {args.transcripts}")
    sizes = sorted(sum(len(str(m.get("content", ""))) for m in p) for p in prompts)
    print(f"  replaying {len(prompts)} recorded prompt(s); prompt size chars "
          f"p50={sizes[len(sizes) // 2]:,} p95={sizes[int(len(sizes) * 0.95)]:,}")
    print(f"  probing {args.levels} for {args.seconds:.0f}s each, max_tokens={args.max_tokens}")
    print("  NOTE: the GPUs are shared with other users, so absolute rates are a lower")
    print("        bound. Read the shape: throughput flattening while latency climbs")
    print("        linearly is saturation.\n")
    print(f"  {'conc':>5}{'rps':>8}{'tok/s':>9}{'mean s':>9}{'p95 s':>9}{'fail':>6}   vllm mid-level")

    results = []
    for concurrency in args.levels:
        outcome = level(concurrency, args.seconds, args.max_tokens, args.timeout, prompts)
        results.append(outcome)
        print(f"  {outcome['concurrency']:>5}{outcome['rps']:>8.2f}{outcome['tok_per_s']:>9.0f}"
              f"{outcome['mean_latency_s']:>9.2f}{outcome['p95_latency_s']:>9.2f}"
              f"{outcome['failures']:>6}   {outcome['vllm']}")

    print("\n  ### reading ###")
    # Judged on generated tokens per second, not requests per second: with real
    # prompts the reply lengths vary, so rps also moves with which prompts happened
    # to be in flight. Decode throughput is what the jobs are waiting on.
    best = max(results, key=lambda r: r["tok_per_s"])
    print(f"  peak decode throughput {best['tok_per_s']:.0f} tok/s at "
          f"concurrency {best['concurrency']}")
    for previous, current in zip(results, results[1:]):
        gain = (current["tok_per_s"] / previous["tok_per_s"] - 1) * 100 \
            if previous["tok_per_s"] else 0.0
        cost = (current["mean_latency_s"] / previous["mean_latency_s"] - 1) * 100 \
            if previous["mean_latency_s"] else 0.0
        verdict = "worth it" if gain > 15 else ("marginal" if gain > 5 else "SATURATED")
        print(f"  {previous['concurrency']:>4} -> {current['concurrency']:<4} "
              f"throughput {gain:+6.1f}%   latency {cost:+6.1f}%   {verdict}")

    with open("logs/concurrency_probe.json", "w", encoding="utf-8") as handle:
        json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "levels": results},
                  handle, indent=2)
    print("\n  written logs/concurrency_probe.json")


if __name__ == "__main__":
    main()
