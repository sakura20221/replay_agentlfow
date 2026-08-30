"""Are the truncated replies unfinished, or stuck repeating?

The distribution says the natural tail dies before 6k while 3,719 replies sit
exactly at the 8,192 cap. That shape is what degeneration looks like, but shape is
an inference -- this reads the text and measures repetition directly.
"""
import collections
import json
import re
from pathlib import Path

lines = Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines()
truncated = []
for line in lines:
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    # The proxy records finish_reason; "length" means it hit the cap.
    if entry.get("finish_reason") == "length" or (entry.get("completion_tokens") or 0) >= 8000:
        truncated.append(entry)

print(f"  truncated exchanges captured in transcripts: {len(truncated)}")
if not truncated:
    print("  (the transcript only covers calls since the last proxy restart)")

def repetition(text: str) -> tuple[float, str]:
    """Share of the reply taken up by its single most repeated line."""
    lines_ = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines_:
        return 0.0, ""
    counts = collections.Counter(lines_)
    line, count = counts.most_common(1)[0]
    return count / len(lines_), line[:80]

examined = 0
degenerate = 0
for entry in truncated[:12]:
    text = entry.get("completion") or ""
    share, worst = repetition(text)
    # A reply whose most common line is a third of it is looping, not reasoning.
    looping = share > 0.30
    degenerate += looping
    examined += 1
    print(f"\n  --- {entry.get('namespace')}  out={entry.get('completion_tokens')} tokens ---")
    print(f"      most repeated line: {share:.0%} of all lines: {worst!r}")
    print(f"      tail: {text[-160:]!r}")
print(f"\n  {degenerate}/{examined} examined replies are dominated by a repeated line")
