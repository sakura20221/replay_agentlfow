"""Confirm the proxy recorded both sides of every stage's LLM call."""
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("logs/transcripts.jsonl").read_text().splitlines() if line.strip()]
calls = [json.loads(line) for line in Path("logs/api_calls.jsonl").read_text().splitlines() if line.strip()]
print(f"  {len(rows)} transcript line(s); namespaces: {sorted({r['namespace'] for r in rows})}")

sample = next(r for r in rows if r["namespace"].endswith("drop"))
sent = sample["messages"][-1]["content"]
print("  sample (probe_live/drop):")
print(f"    input  {len(sent)} chars | strict format present={'MUST end with a line' in sent} "
      f"| ICL example present={'Format example' in sent}")
print(f"    input last 3 lines: {sent.strip().splitlines()[-3:]}")
print(f"    output {len(sample['completion'])} chars | last line="
      f"{sample['completion'].strip().splitlines()[-1]!r}")
print(f"    tokens {sample['prompt_tokens']}/{sample['completion_tokens']} finish={sample['finish_reason']}")

ids = {r["request_id"] for r in rows}
ok = {c["request_id"] for c in calls if c.get("status") == 200}
print(f"  joinable to api_calls.jsonl on request_id: {len(ids & ok)}/{len(ids)}")
print(f"  size on disk: {Path('logs/transcripts.jsonl').stat().st_size / 1024:.1f} KB "
      f"for {len(rows)} call(s) -> {Path('logs/transcripts.jsonl').stat().st_size / max(len(rows),1):.0f} bytes/call")
