#!/usr/bin/env python3
"""How long does correct MBPP code take to run, so the timeout can be justified.

A timeout that is too tight fails legitimate solutions -- and it would not fail them
evenly, since the methods that generate heavier code would lose more. So the number
comes from the distribution of *correct* runs plus a wide margin, not from intuition.

Reference solutions are used because they are known-correct: whatever they take is
the floor any timeout must clear.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import statistics
import time
from pathlib import Path


def _run(source: str, done) -> None:
    start = time.perf_counter()
    try:
        exec(source, {})  # noqa: S102 - mirrors the operator
        done.value = time.perf_counter() - start
    except Exception:  # noqa: BLE001 - a wrong reference solution is still timed
        done.value = time.perf_counter() - start


def timed(source: str, limit: float) -> float | None:
    """Seconds taken, or None if it outlived the limit."""
    elapsed = multiprocessing.Value("d", -1.0)
    process = multiprocessing.Process(target=_run, args=(source, elapsed))
    process.start()
    process.join(limit)
    if process.is_alive():
        process.terminate()
        process.join(3)
        if process.is_alive():
            process.kill()
        return None
    return elapsed.value if elapsed.value >= 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=float, default=60.0,
                        help="hard ceiling while measuring")
    args = parser.parse_args()

    rows = []
    for name in ("mbpp.jsonl", "mbpp_search.jsonl"):
        path = Path("shared/data", name)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    print(f"  {len(rows)} MBPP item(s) with reference solutions")

    times, slow, timedout = [], [], 0
    for row in rows:
        code = row.get("code") or row.get("solution") or ""
        entry = row.get("entry_point")
        tests = row.get("test_list") or []
        if not code or not entry or not tests:
            continue
        harness = (code + "\n\n" + f"candidate = {entry}\n"
                   + "\n".join(re.sub(rf"\b{re.escape(entry)}\b", "candidate", t)
                               for t in tests))
        taken = timed(harness, args.limit)
        if taken is None:
            timedout += 1
            print(f"    reference solution for {entry} exceeded {args.limit}s")
            continue
        times.append(taken)
        if taken > 0.5:
            slow.append((taken, entry))

    times.sort()
    if not times:
        print("  nothing measured")
        return
    def pct(q):
        return times[min(int(len(times) * q), len(times) - 1)]
    print(f"\n  measured {len(times)} run(s); {timedout} exceeded the measuring ceiling")
    print(f"    median {pct(.5) * 1000:.1f} ms")
    print(f"    p99    {pct(.99) * 1000:.1f} ms")
    print(f"    p99.9  {pct(.999) * 1000:.1f} ms")
    print(f"    max    {times[-1] * 1000:.1f} ms")
    print(f"\n  slowest correct solutions:")
    for taken, entry in sorted(slow, reverse=True)[:6]:
        print(f"    {taken:7.2f}s  {entry}")
    print(f"\n  a timeout of {max(5, int(times[-1] * 10) + 1)}s would be "
          f"{max(5, int(times[-1] * 10) + 1) / max(times[-1], 1e-6):.0f}x the slowest "
          f"correct run")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    main()
