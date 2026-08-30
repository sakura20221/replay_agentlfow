"""Who is losing requests to the context ceiling, and how badly.

The proxy can clamp max_tokens but cannot shrink a prompt, so a prompt that alone
approaches max_model_len fails with a hard 400 no matter what. That is not evenly
distributed: it hits whichever method accumulates the most history per call, so
left unhandled it is a confound, not just noise.
"""
import collections
import json
import re

by_ns = collections.Counter()
totals = collections.Counter()
prompt_sizes = collections.defaultdict(list)

for line in open("logs/api_calls.jsonl"):
    try:
        record = json.loads(line)
    except Exception:  # noqa: BLE001
        continue
    ns = record.get("namespace", "?")
    totals[ns] += 1
    error = str(record.get("error") or "")
    if "maximum context length" in error:
        by_ns[ns] += 1
        found = re.search(r"prompt contains at least (\d+) input tokens", error)
        if found:
            prompt_sizes[ns].append(int(found.group(1)))

print(f"  {'method':<20}{'requests':>10}{'ctx-overflow':>14}{'rate':>8}   prompt tokens seen")
for ns, count in by_ns.most_common():
    sizes = prompt_sizes[ns]
    span = f"{min(sizes):,}-{max(sizes):,}" if sizes else "-"
    print(f"  {ns:<20}{totals[ns]:>10,}{count:>14,}{count / max(totals[ns], 1):>7.2%}   {span}")
if not by_ns:
    print("  none")
print(f"\n  total requests logged: {sum(totals.values()):,}")
