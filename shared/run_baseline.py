#!/usr/bin/env python3
"""Single-agent baselines (IO / CoT) on the shared benchmarks.

Two jobs in one script:

* it is the capability floor for the bake-off -- any workflow method that scores
  below CoT on the same executor is not worth building on;
* it is the end-to-end check of the plumbing (proxy -> vLLM -> shared scoring)
  and needs no per-repo shim, so it can run before the shims exist.

Every run records the diagnostics that decide whether a score is meaningful at
all on an 8B executor: truncation rate and answer-extraction failure rate.

    python run_baseline.py --method cot --dataset math --limit 50
    python run_baseline.py --method cot --all --concurrency 48
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bench

PROXY = "http://127.0.0.1:18080"
MODEL = "Qwen/Qwen3-8B"
RESULTS = Path(__file__).resolve().parent / "results"

# The instruction differs only in whether reasoning is requested; the task text
# itself comes from bench.question_text so every method sees the same problem.
SYSTEM = {
    "io": "Answer the question directly. Do not explain.",
    "cot": "Reason step by step, then state the final answer.",
}

ANSWER_FORMAT = {
    "math": "End your reply with the final answer inside \\boxed{}.",
    "amc": "End your reply with the final answer inside \\boxed{}.",
    "mbpp": "Return self-contained Python code defining the requested entry point inside a ```python code block.",
    "drop": "End your reply with 'Answer: <answer>', using a concise span, number, date, or list as appropriate.",
    "mmlu_pro": "End your reply with 'Answer: (X)' where X is the option letter.",
}


def call(namespace: str, system: str, user: str, max_tokens: int, timeout: float = 600.0) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{PROXY}/{namespace}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_one(args: tuple[str, str, dict]) -> dict:
    method, dataset, row = args
    namespace = f"test/{method}_{dataset}"
    system = SYSTEM[method] + " " + ANSWER_FORMAT[dataset]
    user = bench.question_text(dataset, row)
    started = time.time()
    try:
        response = call(namespace, system, user, bench.MAX_TOKENS[dataset])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return {"uid": row["uid"], "score": 0.0, "error": f"{type(exc).__name__}: {exc}",
                "latency": time.time() - started, "truncated": False, "no_answer": True,
                "completion_tokens": 0}

    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = response.get("usage") or {}
    value, extracted = bench.score(dataset, row, content)
    return {
        "uid": row["uid"],
        "score": value,
        "extracted": str(extracted)[:120],
        "truncated": choice.get("finish_reason") == "length",
        "no_answer": not str(extracted).strip(),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency": time.time() - started,
    }


def run_dataset(method: str, dataset: str, limit: int | None, concurrency: int) -> dict:
    rows = list(bench.load(dataset))
    if limit:
        rows = rows[:limit]
    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(run_one, [(method, dataset, row) for row in rows]))
    wall = time.time() - started

    scores = [r["score"] for r in records]
    errors = sum(1 for r in records if r.get("error"))
    summary = {
        "method": method,
        "dataset": dataset,
        "metric": bench.metric_name(dataset),
        "n": len(records),
        "score": round(100 * statistics.fmean(scores), 2) if scores else 0.0,
        "truncated_pct": round(100 * sum(r["truncated"] for r in records) / max(len(records), 1), 2),
        "no_answer_pct": round(100 * sum(r["no_answer"] for r in records) / max(len(records), 1), 2),
        "error_pct": round(100 * errors / max(len(records), 1), 2),
        "avg_completion_tokens": round(statistics.fmean([r["completion_tokens"] for r in records]), 1),
        "wall_seconds": round(wall, 1),
        "req_per_s": round(len(records) / wall, 2) if wall else 0.0,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{method}_{dataset}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(SYSTEM), default="cot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=bench.DATASETS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N items (smoke runs)")
    parser.add_argument("--concurrency", type=int, default=48)
    args = parser.parse_args()

    targets = list(bench.DATASETS) if args.all else [args.dataset]
    header = f"{'dataset':10s} {'metric':9s} {'n':>5s} {'score':>7s} {'trunc%':>7s} {'noans%':>7s} {'err%':>6s} {'tok':>7s} {'req/s':>6s}"
    print(header)
    print("-" * len(header))
    summaries = []
    for dataset in targets:
        summary = run_dataset(args.method, dataset, args.limit, args.concurrency)
        summaries.append(summary)
        print(f"{summary['dataset']:10s} {summary['metric']:9s} {summary['n']:5d} "
              f"{summary['score']:7.2f} {summary['truncated_pct']:7.2f} {summary['no_answer_pct']:7.2f} "
              f"{summary['error_pct']:6.2f} {summary['avg_completion_tokens']:7.1f} {summary['req_per_s']:6.2f}")

    if len(summaries) > 1:
        macro = statistics.fmean(s["score"] for s in summaries)
        print("-" * len(header))
        print(f"{'MACRO':10s} {'mean':9s} {sum(s['n'] for s in summaries):5d} {macro:7.2f}")

    worst = max(summaries, key=lambda s: s["truncated_pct"])
    if worst["truncated_pct"] > 10:
        print(f"\nWARNING: {worst['dataset']} truncated {worst['truncated_pct']}% of replies at "
              f"max_tokens={bench.MAX_TOKENS[worst['dataset']]}; scores there understate the model.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
