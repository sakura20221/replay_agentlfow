"""How close do prompts actually run to the window, per method?

Deciding whether a bigger window fixes the overflow needs the distribution, not the
single failing value: 32,706 is where requests *hit the wall*, which says nothing
about how large they would have grown with more room. If the mass sits far below
the ceiling with a thin tail, a larger window absorbs the tail. If prompts pile up
against the ceiling, the wall just moves.
"""
import collections
import json
from pathlib import Path

WINDOW = 32768
by_method = collections.defaultdict(list)

for line in Path("logs/api_calls.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    tokens = record.get("prompt_tokens")
    if isinstance(tokens, int) and tokens > 0:
        by_method[record.get("namespace", "?")].append(tokens)


def pct(values, q):
    values = sorted(values)
    return values[min(int(len(values) * q), len(values) - 1)]


print(f"  window = {WINDOW:,}")
print(f"  {'method':<20}{'calls':>9}{'median':>9}{'p90':>9}{'p99':>9}{'p99.9':>9}{'max':>9}"
      f"{'within 2k of wall':>19}")
for method, values in sorted(by_method.items(), key=lambda kv: -len(kv[1])):
    if len(values) < 100:
        continue
    near = sum(1 for v in values if v > WINDOW - 2000)
    print(f"  {method:<20}{len(values):>9,}{pct(values, .5):>9,}{pct(values, .9):>9,}"
          f"{pct(values, .99):>9,}{pct(values, .999):>9,}{max(values):>9,}"
          f"{near:>13,} ({near / len(values):.3%})")

everything = [v for vs in by_method.values() for v in vs]
print(f"\n  all calls: {len(everything):,}")
for threshold in (8192, 16384, 24576, 30768, 32000, 32700):
    over = sum(1 for v in everything if v > threshold)
    print(f"    prompts over {threshold:>6,}: {over:>7,} ({over / len(everything):.3%})")
