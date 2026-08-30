#!/usr/bin/env python3
"""Stop every sweep driver and method process, and nothing else.

Written as a file rather than a shell one-liner because `pkill -f` and
`ps | grep` match the command line of the very ssh session issuing them: three
times today that killed my own connection mid-operation. Matching is done here on
the resolved executable path plus argv, with this process and its ancestors
excluded explicitly.

Leaves the proxy, the vLLM servers and the watchdog alone -- they are shared
infrastructure, not jobs.
"""
from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVS = str(ROOT / "envs")

# argv fragments that identify a job or a driver.
JOB_MARKERS = ("sweep.py", "examples/maas/optimize.py", "experiments/run_shared.py",
               "Experiments/run_shared.py", "run.py --dataset")
# Never touch these.
KEEP = ("vllm_proxy.py", "vllm serve", "vllm.entrypoints", "watchdog.py")


def ancestors(pid: int) -> set[int]:
    chain = set()
    current = pid
    while current > 1:
        chain.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text()
            current = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually signal (default: list)")
    parser.add_argument("--signal", default="TERM", choices=["TERM", "KILL"])
    args = parser.parse_args()

    protect = ancestors(os.getpid())
    victims = []
    for entry in sorted(Path("/proc").iterdir()):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in protect:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        if not argv.strip():
            continue
        if ENVS not in argv and "sweep.py" not in argv:
            continue
        if any(keep in argv for keep in KEEP):
            continue
        if not any(marker in argv for marker in JOB_MARKERS):
            continue
        victims.append((pid, argv.strip()[:110]))

    print(f"  {len(victims)} job/driver process(es) matched"
          f"{' -- signalling ' + args.signal if args.apply else ' (dry run)'}")
    for pid, argv in victims:
        print(f"    {pid:>9}  {argv}")
        if args.apply:
            try:
                os.kill(pid, signal.SIGKILL if args.signal == "KILL" else signal.SIGTERM)
            except ProcessLookupError:
                pass
    if not args.apply:
        print("\n  nothing signalled. Re-run with --apply.")


if __name__ == "__main__":
    main()
