"""Did the context overflows land in a search phase or a held-out test phase?

It matters: a lost search call costs one training sample, a lost test call changes
a reported score. The proxy namespace cannot answer this -- every method posts both
phases to the same `train/<method>` URL -- so the question is settled by matching
each failure's timestamp against the windows in which test phases actually ran.
"""
import datetime
import json
import os
from pathlib import Path

RUNS = Path(os.getenv("SWEEP_RUNS", "runs_icl"))

# Test-phase windows, taken from each job's own test.log: first and last mtime we
# can establish. Only MaAS-family jobs have a separate test.log; for the methods
# that train and test in one process the window is the whole job, which is stated
# rather than guessed at.
windows = []
for test_log in sorted(RUNS.glob("*/*/repeat*/test.log")):
    job = f"{test_log.parents[2].name}/{test_log.parents[1].name}"
    stat = test_log.stat()
    # st_mtime is the end; the start is approximated by the search phase's end.
    seconds_file = test_log.parent / "search.seconds"
    start = None
    if seconds_file.exists():
        start = datetime.datetime.fromtimestamp(seconds_file.stat().st_mtime,
                                                datetime.timezone.utc)
    end = datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc)
    if start:
        windows.append((job, start, end))

print(f"  {len(windows)} test-phase window(s) established:")
for job, start, end in windows:
    print(f"    {job:<22} {start:%H:%M} -> {end:%H:%M}")

overflows = []
for line in Path("logs/api_calls.jsonl").read_text(errors="replace").splitlines():
    if "maximum context length" not in line:
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("timestamp"):
        overflows.append(record)

in_test = 0
by_method = {}
for record in overflows:
    when = datetime.datetime.fromisoformat(record["timestamp"])
    method = (record.get("namespace") or "").split("/")[-1]
    hit = next((j for j, s, e in windows if s <= when <= e and j.startswith(method)), None)
    by_method.setdefault(method, {"total": 0, "in_test": 0})
    by_method[method]["total"] += 1
    if hit:
        by_method[method]["in_test"] += 1
        in_test += 1

print(f"\n  {len(overflows)} overflow(s); {in_test} fall inside a known test-phase window")
for method, counts in sorted(by_method.items(), key=lambda kv: -kv[1]["total"]):
    print(f"    {method:<12} {counts['total']:>4} total, {counts['in_test']:>3} during a test phase")
print("\n  Methods with no separate test.log (card, gdesigner, masrouter, flowbank)")
print("  train and evaluate in one process, so for those this test cannot separate")
print("  the phases at all -- stated rather than assumed.")
