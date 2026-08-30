"""Truncation rate per method -- a confound that does not fall equally.

A reply cut off at max_tokens scores zero for a reason unrelated to the method, and
methods do not truncate at equal rates: a workflow that decomposes a task emits
short per-node replies, while one long chain of thought hits the cap often. The
proxy recovers truncated replies by asking for the final answer in a short
follow-up, so what matters for the table is the rate *and* whether recovery worked.
"""
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:18080/stats", timeout=10) as response:
    stats = json.loads(response.read())

print(f"  {'method':<20}{'requests':>10}{'truncated':>11}{'rate':>8}"
      f"{'recovered':>11}{'empty':>8}")
rows = []
for name, counters in stats["by_namespace"].items():
    requests = counters.get("requests", 0)
    if requests < 100:
        continue
    truncated = counters.get("truncated", 0)
    rows.append((truncated / max(requests, 1), name, requests, truncated,
                 counters.get("recovered", 0), counters.get("empty_content", 0)))
for rate, name, requests, truncated, recovered, empty in sorted(rows, reverse=True):
    print(f"  {name:<20}{requests:>10,}{truncated:>11,}{rate:>7.1%}"
          f"{recovered:>11,}{empty:>8,}")

totals = stats["totals"]
requests = totals.get("requests", 1)
print(f"\n  overall: {requests:,} requests, {totals.get('truncated', 0):,} truncated "
      f"({totals.get('truncated', 0) / requests:.1%}), "
      f"{totals.get('recovered', 0):,} recovered, "
      f"{totals.get('empty_content', 0):,} empty")
