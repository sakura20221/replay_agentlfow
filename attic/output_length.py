"""How long do replies actually run, and what cap would make truncation negligible?

Two populations have to be separated before any cap can be justified:

* replies that were merely long -- a bigger cap finishes them;
* replies that had degenerated into repetition -- they run to *any* cap, so raising
  it converts a truncated loop into a longer truncated loop and nothing else.

Only the first population responds to the number this script is asked to produce.
"""
import collections
import json
import re
from pathlib import Path

CAP = 8192
lengths = collections.defaultdict(list)
truncated_ids = []

for line in Path("logs/api_calls.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    n = record.get("completion_tokens")
    if not isinstance(n, int) or n <= 0:
        continue
    if record.get("truncated"):
        truncated_ids.append(record.get("request_id"))
    else:
        lengths[record.get("namespace", "?")].append(n)


def pct(values, q):
    values = sorted(values)
    return values[min(int(len(values) * q), len(values) - 1)]


everything = [v for vs in lengths.values() for v in vs]
print(f"  completed (untruncated) replies: {len(everything):,}   truncated: {len(truncated_ids):,}"
      f"   ({len(truncated_ids) / (len(everything) + len(truncated_ids)):.2%})")
print(f"\n  {'method':<20}{'n':>9}{'median':>8}{'p90':>8}{'p99':>8}{'p99.9':>9}{'max':>8}")
for method, values in sorted(lengths.items(), key=lambda kv: -len(kv[1])):
    if len(values) < 100:
        continue
    print(f"  {method:<20}{len(values):>9,}{pct(values, .5):>8}{pct(values, .9):>8}"
          f"{pct(values, .99):>8}{pct(values, .999):>9}{max(values):>8}")

print(f"\n  overall percentiles: p99={pct(everything, .99):,}  p99.9={pct(everything, .999):,}"
      f"  p99.99={pct(everything, .9999):,}  max={max(everything):,}")
over = lambda t: sum(1 for v in everything if v > t)
for threshold in (2048, 4096, 6144, 8000):
    print(f"    untruncated replies longer than {threshold:>5,}: {over(threshold):>6,} "
          f"({over(threshold) / len(everything):.3%})")
