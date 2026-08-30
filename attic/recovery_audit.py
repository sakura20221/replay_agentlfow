"""Did truncation recovery actually recover, and would a bigger window help?

Two things to establish, both of which I previously asserted without checking:

1. I reported "3,033 truncated, 3,033 recovered, 100%". The `recovered` flag is set
   by the proxy before it knows the continuation succeeded, so the claim needs the
   continuation's own outcome, not the flag.
2. Whether raising the window from 32,768 to 40,960 removes the overflow. The
   binding constraint for a continuation is prompt + generated + follow-up, so the
   question is how many first-attempt prompts sit above W - 8192 - 256.
"""
import json
from pathlib import Path

records = []
for line in Path("logs/api_calls.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        continue

truncated = [r for r in records if r.get("truncated")]
failed_continuation = [r for r in truncated if r.get("status") == 502 or r.get("continuation_error")]
print(f"  truncated calls: {len(truncated):,}")
print(f"  of those, the continuation itself failed: {len(failed_continuation):,} "
      f"({len(failed_continuation) / max(len(truncated), 1):.1%})")
reasons = {}
for r in failed_continuation:
    key = str(r.get("continuation_error") or r.get("error"))[:90]
    reasons[key] = reasons.get(key, 0) + 1
for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:3]:
    print(f"    [{count}x] {reason}")

# First-attempt prompt sizes: exclude records whose prompt_tokens was overwritten by
# a continuation call (those carry truncated=True and an inflated prompt).
first = [r.get("prompt_tokens") for r in records
         if isinstance(r.get("prompt_tokens"), int) and r.get("prompt_tokens") > 0
         and not r.get("truncated")]
print(f"\n  first-attempt prompts measured: {len(first):,}   max: {max(first):,}")
for window in (32768, 40960):
    # A plain call needs prompt + 16; a continuation needs prompt + 8192 + 256.
    plain = sum(1 for p in first if p + 16 > window)
    with_continuation = sum(1 for p in first if p + 8192 + 256 > window)
    print(f"  window {window:,}: {plain:>5} prompt(s) too big even for a 16-token reply"
          f"   {with_continuation:>5} too big to survive a truncation continuation"
          f"  ({with_continuation / len(first):.3%})")
