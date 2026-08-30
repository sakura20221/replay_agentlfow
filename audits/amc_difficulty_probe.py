#!/usr/bin/env python3
"""Measure which AMC evaluation set is harder for the actual model.

Decision support for 2026-08-24: ours is AI-MO/aimo-validation-amc (83 items,
the 2022-2023 AMC12 papers with numeric answers); FlowBank ships its own
amc_validate/amc_test (165+655) whose rows carry MATH-style level/type fields.
Which one is harder cannot be read off metadata -- the honest measure is the
same model, the same plain prompt, the same scorer, on a sample of each.

Single plain completion per problem, no workflow: the absolute numbers are not
comparable to any method's score, only the DIFFERENCE between the two sets
matters. Both sides get identical treatment, scored by shared/bench's math
scorer (sympy equivalence), gold for FlowBank rows = last \\boxed of the
shipped solution (their own benchmark extracts gold the same way).

    envs/maas/bin/python audits/amc_difficulty_probe.py --n 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402

PROXY = "http://127.0.0.1:18080/probe/amc/v1/chat/completions"
PROMPT = ("Solve the following problem. Show your reasoning briefly, then give "
          "your final answer in \\boxed{}.\n\n")


def boxed_tail(solution: str) -> str:
    """Last \\boxed{...} of a MATH-style solution, balanced-brace scan."""
    index = solution.rfind("\\boxed")
    if index < 0:
        return ""
    tail = solution[index + len("\\boxed"):]
    if not tail.startswith("{"):
        return tail.strip().split("$")[0].strip()
    depth = 0
    for position, char in enumerate(tail):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return tail[1:position]
    return ""


def ask(problem: str) -> str:
    body = json.dumps({
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": PROMPT + problem}],
        "max_tokens": 4096,
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        PROXY, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read())
        return data["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001 - a failed call scores 0, recorded
        return f"[probe request failed: {type(exc).__name__}]"


def run(label: str, items: list[tuple[str, str]], jobs: int) -> None:
    """items: (problem, gold). Prints per-set accuracy and the misses."""
    with ThreadPoolExecutor(jobs) as pool:
        replies = list(pool.map(ask, (problem for problem, _ in items)))
    scores = []
    for (problem, gold), reply in zip(items, replies):
        value, extracted = bench.score("math", {"answer": gold, "solution": ""}, reply)
        scores.append(value)
        if value < 0.999:
            print(f"    miss gold={gold[:24]!r:<28} got={extracted[:36]!r}")
    print(f"  {label}: {sum(scores):.0f}/{len(scores)} = {sum(scores) / len(scores):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    flowbank = [json.loads(l) for l in
                (ROOT / "third_party/flowbank/datasets/amc_test.jsonl").open(encoding="utf-8")]
    with_gold = [(r["problem"], boxed_tail(r["solution"]))
                 for r in flowbank if boxed_tail(r["solution"])]
    print(f"flowbank amc_test: {len(flowbank)} rows, {len(with_gold)} with \\boxed gold")
    from collections import Counter
    print(f"  level: {dict(Counter(str(r.get('level')) for r in flowbank))}")
    print(f"  type:  {dict(Counter(str(r.get('type')) for r in flowbank))}")

    ours = [json.loads(l) for l in (ROOT / "shared/data/amc.jsonl").open(encoding="utf-8")]
    print(f"ours amc.jsonl: {len(ours)} rows (AI-MO aimo-validation-amc)")

    print(f"\nprobing {args.n} of each with a single plain completion ...")
    run("flowbank amc_test", rng.sample(with_gold, min(args.n, len(with_gold))), args.jobs)
    run("ours amc-83      ", [(r["problem"], str(r["answer"]))
                              for r in rng.sample(ours, min(args.n, len(ours)))], args.jobs)
    print("\nonly the gap between the two lines matters; neither number is a "
          "method score.")


if __name__ == "__main__":
    main()
