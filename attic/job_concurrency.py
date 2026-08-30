#!/usr/bin/env python3
"""How many requests does one job actually keep in flight?

`--jobs` is a number of processes; what the serving side sees is a number of
concurrent requests, and the ratio between them is a property of each method --
AFlow evaluates a whole validation split at once, MasRouter dispatches a batch of
16, the G-Designer family sends four at a time. So the right `--jobs` cannot be
reasoned about, only measured.

Computed by sweeping the start/end times in logs/api_calls.jsonl: each record
carries a timestamp and a latency, which gives an interval, and the number of
overlapping intervals at any instant is the concurrency. Reported per namespace
so a method that contributes almost nothing is visible as such.
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path: Path, namespace_filter: str | None) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        stamp = record.get("timestamp")
        latency = record.get("latency_ms")
        if not stamp or latency is None:
            continue
        namespace = record.get("namespace") or "unspecified"
        if namespace_filter and namespace_filter not in namespace:
            continue
        try:
            end = datetime.fromisoformat(stamp).timestamp()
        except (ValueError, TypeError):
            continue
        # The timestamp is written when the record is created, at request start.
        intervals[namespace].append((end, end + float(latency) / 1000.0))
    return intervals


def concurrency_profile(intervals: list[tuple[float, float]]) -> tuple[float, int, float]:
    """(time-weighted mean, peak, span_seconds) of overlapping intervals."""
    if not intervals:
        return 0.0, 0, 0.0
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    current = 0
    weighted = 0.0
    peak = 0
    previous = events[0][0]
    for moment, delta in events:
        weighted += current * (moment - previous)
        previous = moment
        current += delta
        peak = max(peak, current)
    span = events[-1][0] - events[0][0]
    return (weighted / span if span else 0.0), peak, span


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="logs/api_calls.jsonl")
    parser.add_argument("--namespace", default=None, help="substring filter")
    parser.add_argument("--target", type=int, default=128,
                        help="in-flight requests the serving side can absorb")
    args = parser.parse_args()

    intervals = load(ROOT / args.log, args.namespace)
    if not intervals:
        raise SystemExit("no usable records")

    print(f"  {'namespace':<26}{'calls':>9}{'mean in flight':>16}{'peak':>7}{'hours':>8}")
    per_job = []
    for namespace, spans in sorted(intervals.items(),
                                   key=lambda kv: -len(kv[1])):
        mean, peak, span = concurrency_profile(spans)
        print(f"  {namespace:<26}{len(spans):>9,}{mean:>16.1f}{peak:>7}{span / 3600:>8.1f}")
        # A namespace is one method, and during the sweep several of its jobs ran
        # at once, so this is not yet per-job -- but the ranking is what matters
        # for deciding which method sets the pace.
        per_job.append((namespace, mean))

    # The number that actually decides --jobs: every method's requests on one
    # timeline. Summing the per-namespace means would overcount, because the
    # namespaces' spans only partly overlap.
    combined = [span for namespace, spans in intervals.items()
                for span in spans if not namespace.startswith(("probe/", "starvation"))]
    mean, peak, span = concurrency_profile(combined)
    print(f"\n  ALL METHODS ON ONE TIMELINE: mean {mean:.1f} in flight, peak {peak}, "
          f"over {span / 3600:.1f}h")

    # How much of the time the serving side was actually kept busy. A run whose
    # mean sits far below its peak is not throughput-limited; it is waiting on its
    # clients, and more of them would help.
    if combined:
        busy_fraction = mean / args.target
        print(f"  that is {busy_fraction:.0%} of the {args.target}-in-flight ceiling "
              f"measured by concurrency_probe.py")
        if busy_fraction < 0.6:
            print(f"  -> room to raise --jobs by roughly {1 / max(busy_fraction, 0.05):.1f}x")
        else:
            print("  -> already near the ceiling; more jobs would queue, not speed up")


if __name__ == "__main__":
    main()
