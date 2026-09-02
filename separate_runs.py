#!/usr/bin/env python3
"""Move an earlier protocol's artefacts aside so a new run cannot mix with them.

Any change to prompts, data, sampling, or search-time grading creates a different
protocol. Results from two protocols must not appear in the same table. Two
mechanisms keep them apart:

* this script, which relocates every artefact the seven methods write, so the new
  run starts from empty directories;
* the protocol stamp (bench.protocol_fingerprint, written per job by sweep.py and
  checked by collect.py), which catches anything this list misses.

The second exists because the first depends on me having enumerated all the
places nine method entries write to, across seven repos, which is exactly the
kind of list that turns out to be incomplete.

**Move, never delete.** The old run cost days of GPU time and is still the only
evidence for several conclusions -- for instance the measured budget asymmetry
(FlowBank 10.5M tokens vs CARD 313K on DROP). Wiping it to make room for a new
table would destroy data that no longer takes any GPU time to keep.

    python separate_runs.py                       # dry run: what would move
    python separate_runs.py --apply --label pre_icl
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"

# Every location a method writes results, checkpoints or searched workflows to.
# Globs are relative to ROOT. Grouped by what produces them so a future addition
# has an obvious home.
ARTEFACTS = (
    # The sweep's own logs, status files and per-job instrumentation.
    "runs",
    # AFlow and FlowBank: searched workflow trees plus their score histories.
    "third_party/aflow/workspace/SHARED_*",
    "third_party/flowbank/DiverseFlow/workspace/SHARED_*",
    # FlowBank stages 2-3: selector data and training logs.
    "third_party/flowbank/experiments",
    "third_party/flowbank/data/shared_*_full",
    # MaAS and DAAO: controller checkpoints and per-round operator pools.
    "third_party/maas/maas/ext/maas/scripts/optimized/SHARED_*",
    "third_party/daao/daao/ext/maas/scripts/optimized/SHARED_*",
    # MasRouter: the trained router, and its own per-run logs (the accuracy the
    # collector parses is printed there, so they are results, not diagnostics).
    "third_party/masrouter/**/*_router_epoch*.pth",
    "third_party/masrouter/result",
    "third_party/masrouter/logs",
    # G-Designer family: the per-item records the accuracy is recomputed from.
    # Their runner *appends*, so a leftover file would blend two protocols'
    # records into one average.
    "third_party/gdesigner/result",
    "third_party/card/result",
    # Proxy accounting. Kept with the run it describes: the token counts and
    # truncation rates belong to those prompts, not to the next protocol's.
    "logs/api_calls.jsonl",
    "logs/transcripts.jsonl",
    # The driver's own stdout. Named individually rather than as logs/*: the proxy
    # and both vLLM servers are still running and still appending to their logs in
    # this directory, and moving a file an open fd points at leaves the process
    # writing into the archive.
    "logs/sweep_*.log",
)

# Process patterns that mean a run is still live. Moving a directory out from
# under a running job produces a half-finished archive and a job writing into a
# path that no longer exists -- worse than either outcome alone.
LIVE_PATTERNS = ("sweep.py", "run_shared.py", "optimize.py", "run.py --dataset",
                 "flowbank_pipeline.py", "aflow_test.py")


def running_jobs() -> list[str]:
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001 - absence of ps must not be read as "nothing running"
        return ["could not run ps: cannot prove the sweep is stopped"]
    hits = []
    for line in out.splitlines():
        if "separate_runs.py" in line:
            continue
        for pattern in LIVE_PATTERNS:
            if pattern in line:
                hits.append(line.strip()[:120])
                break
    return hits


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def targets(extra_run_dirs: list[str] | None = None) -> list[Path]:
    found: list[Path] = []
    for pattern in ARTEFACTS:
        if any(ch in pattern for ch in "*?["):
            found.extend(sorted(ROOT.glob(pattern)))
        else:
            path = ROOT / pattern
            if path.exists():
                found.append(path)
    for value in extra_run_dirs or []:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if path == ROOT or ROOT.resolve() not in path.parents:
            raise SystemExit(f"refusing run directory outside the repository: {path}")
        if path.exists():
            found.append(path)
    # A caller may explicitly pass the default `runs`, which is already listed.
    return list(dict.fromkeys(found))


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    parser.add_argument("--label", default=f"protocol_{time.strftime('%Y%m%d_%H%M')}",
                        help="archive subdirectory name")
    parser.add_argument("--force", action="store_true",
                        help="move even though a job looks live (do not use)")
    parser.add_argument("--runs-dir", action="append", default=[],
                        help="additional run directory used by the current pipeline")
    args = parser.parse_args()

    live = running_jobs()
    if live and not args.force:
        print("  refusing: these look like live jobs --")
        for line in live:
            print(f"    {line}")
        raise SystemExit("stop the sweep first, or pass --force if these are unrelated")

    destination = ARCHIVE / args.label
    found = targets(args.runs_dir)
    if not found:
        print("  nothing to archive: all artefact locations are already empty")
        return

    total = 0
    print(f"  {'would move' if not args.apply else 'moving'} -> {destination}")
    for path in found:
        size = size_of(path)
        total += size
        relative = path.relative_to(ROOT)
        print(f"    {human(size):>9}  {relative}")
        if args.apply:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    print(f"  {len(found)} path(s), {human(total)} total")

    if args.apply:
        (destination / "WHY.txt").write_text(
            "Artefacts from a previous experiment protocol. They are isolated because\n"
            "different prompts, data, sampling, or search-time grading cannot share a\n"
            "result table. Kept for audit and measured-budget evidence.\n"
            f"\narchived {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8")
        print(f"  wrote {destination / 'WHY.txt'}")
        print("  the new run starts from empty artefact directories")
    else:
        print("  dry run: nothing moved. Re-run with --apply to do it.")


if __name__ == "__main__":
    main()
