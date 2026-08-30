"""Did the probe itself lose calls to the context window?

My sweep scripts record a failed call as an empty reply, which would then be
counted as "no answer produced" -- an error would look like a quality result. So
the prompt sizes and the error count have to be checked before the tables mean
anything.
"""
import json
import re
from pathlib import Path

WINDOW = 32768
PROBE_MAX_TOKENS = 2048

sizes = []
seen = set()
for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (entry.get("completion_tokens") or 0) < 8000:
        continue
    messages = entry.get("messages") or []
    if not messages:
        continue
    key = json.dumps(messages, ensure_ascii=False)[:400]
    if key in seen:
        continue
    seen.add(key)
    sizes.append(entry.get("prompt_tokens") or 0)
    if len(sizes) >= 30:
        break

print(f"  the 30 replayed prompts, prompt_tokens as recorded:")
print(f"    min {min(sizes):,}  median {sorted(sizes)[len(sizes) // 2]:,}  max {max(sizes):,}")
risky = [s for s in sizes if s + PROBE_MAX_TOKENS > WINDOW]
print(f"    prompts that would exceed the window at max_tokens={PROBE_MAX_TOKENS}: {len(risky)}")
if risky:
    print(f"      sizes: {sorted(risky)}")
print(f"    headroom of the largest: {WINDOW - max(sizes) - PROBE_MAX_TOKENS:,} tokens")
