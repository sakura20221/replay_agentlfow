#!/usr/bin/env python3
"""Watch a running sweep and act on the one failure nothing else survives.

Written because "I'll keep an eye on it" is not something an assistant that only
runs when messaged can honestly promise. This runs on the box, independent of any
session, and leaves a record that can be read later rather than reconstructed.

What it does autonomously, and only this:

* **Restarts the proxy if it stops answering.** Every job of every method goes
  through it, so a dead proxy fails the whole matrix within minutes, and bringing
  it back is pure recovery -- the counters reset, but no artefact is touched.

Everything else is recorded, not acted on. Restarting a vLLM instance on a shared
GPU can land on memory a neighbour has since taken; stopping the sweep because a
job failed throws away work a human might want to keep. Those are judgement calls
and they get surfaced, not guessed at.

Nothing here holds a credential: this box is a shared account, so a notification
key left on it would be readable by eight other people.

    python watchdog.py --interval 300
    python watchdog.py --once          # one pass, for checking the checks
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.getenv("SWEEP_RUNS", str(ROOT / "runs")))
LOG = ROOT / "logs" / "watchdog.jsonl"
STATUS = ROOT / "logs" / "watchdog_status.txt"
PROXY = os.getenv("PROXY_URL", "http://127.0.0.1:18080")
# Below this, the run is close enough to filling a disk that nine people share
# for it to be worth saying so loudly.
DISK_FLOOR_GB = float(os.getenv("WATCHDOG_DISK_FLOOR_GB", "25"))


def shell(command: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def proxy_stats() -> dict | None:
    try:
        with urllib.request.urlopen(f"{PROXY}/stats", timeout=10) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def restart_proxy() -> str:
    """Bring the proxy back in its own tmux session, as it was launched."""
    transcript = os.getenv("PROXY_TRANSCRIPT_PATH", "logs/transcripts.jsonl")
    cap = os.getenv("PROXY_TRANSCRIPT_MAX_BYTES", str(20 * 1024 ** 3))
    shell("tmux kill-session -t sml_proxy 2>/dev/null")
    time.sleep(2)
    shell(f"tmux new-session -d -s sml_proxy -c {ROOT} "
          f"'PROXY_TRANSCRIPT_PATH={transcript} PROXY_TRANSCRIPT_MAX_BYTES={cap} "
          f"{ROOT}/envs/vllm/bin/python vllm_proxy.py >> logs/proxy.log 2>&1'")
    time.sleep(8)
    return "recovered" if proxy_stats() else "restart failed"


# A job is called stalled when BOTH are true: its log has not grown for this long,
# and its namespace has completed no request in that window. Either alone gives
# false positives -- a long optimiser step is quiet in the log while still making
# calls, and a job doing local work between calls is quiet on the wire while still
# writing. Requiring both is what distinguishes "slow" from "stopped".
STALL_MINUTES = int(os.getenv("WATCHDOG_STALL_MINUTES", "45"))


def namespace_last_seen() -> dict[str, float]:
    """When each namespace last completed a request, from the proxy's own log.

    Read from the tail rather than the whole file: api_calls.jsonl reaches millions
    of lines over a sweep, and only the recent end can say anything about now.
    """
    path = Path(os.getenv("PROXY_LOG_PATH", "logs/api_calls.jsonl"))
    if not path.exists():
        return {}
    latest: dict[str, float] = {}
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            window = min(handle.tell(), 8 * 1024 ** 2)
            handle.seek(-window, os.SEEK_END)
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return {}
    for line in chunk.splitlines()[1:]:  # first line may be partial
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        namespace = record.get("namespace")
        stamp = record.get("timestamp")
        if not namespace or not stamp:
            continue
        try:
            moment = datetime.fromisoformat(stamp).timestamp()
        except (ValueError, TypeError):
            continue
        if moment > latest.get(namespace, 0.0):
            latest[namespace] = moment
    return latest


_MISCORE_EVERY = 6          # cycles; at --interval 600 this is hourly
_miscore_counter = {"n": 0}


def miscore_scan() -> list[str]:
    """Hourly incremental correct-but-zero over fresh per-item records.

    The mmlu_pro letter-extractor defect ran a full day before a human noticed a
    low score and asked; every artefact needed for detection was on disk the
    whole time. This closes that gap: re-grade recent records with the current
    scorer and flag any cell where "answered right, scored zero" exceeds noise.
    drop is excluded (its word-form false-positives would cry wolf: the official
    F1 keeps 0.2-0.4% there by design) and so is mbpp (re-grading executes the
    test suites -- too heavy for a watchdog cycle). Thresholded, because the
    math/amc prefilter also catches genuine sign errors (gold 4/3 is a substring
    of the wrong answer -4/3).
    """
    _miscore_counter["n"] += 1
    if _miscore_counter["n"] % _MISCORE_EVERY != 1:
        return []
    since = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 3 * 3600))
    raw = shell(f"{ROOT}/envs/maas/bin/python {ROOT}/audits/correct_but_zero.py "
                f"--datasets mmlu_pro math amc --tag v5 --since '{since}'",
                timeout=900)
    alarms = []
    for line in raw.splitlines():
        parts = line.split()
        # data rows look like:  cell  n  full  partial  zero  suspects  fresh  stored
        if len(parts) == 8 and "/" in parts[0]:
            try:
                n = int(parts[1].replace(",", ""))
                suspects = int(parts[5].replace(",", ""))
            except ValueError:
                continue
            if n >= 50 and suspects >= 3 and suspects / n > 0.01:
                alarms.append(f"{parts[0]}: {suspects}/{n} answered-right-scored-zero")
    return alarms


def contamination() -> list[dict]:
    """Cross-dataset wording in the prompts actually sent, per cell.

    Run every cycle rather than once by hand, because the cells do not all start at
    the same time: the sweep orders datasets by size, so MATH and MBPP begin hours
    after DROP and MMLU-Pro. A one-off inspection can only see the cells that
    happen to be running when it is done -- and the two contamination bugs found so
    far were both invisible to install-time checks and only showed up in live
    traffic. Only the tail of the transcript is scanned, so the cost stays flat as
    the file grows.
    """
    tool = ROOT / "audits" / "live_contamination.py"
    if not tool.exists():
        return []
    raw = shell(f"{ROOT}/envs/tools/bin/python {tool} --json --tail-mb 48", timeout=180)
    if not raw.startswith("{"):
        return []
    try:
        return json.loads(raw).get("findings", [])
    except json.JSONDecodeError:
        return []


def _ancestors(pid: int) -> set[int]:
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


def find_job_pids(command: str) -> list[int]:
    """PIDs running this job's command, matched from /proc rather than pgrep.

    `pgrep -f` matches the command line of the process that invoked it, so its
    first hit is its own shell -- it reported "1 pid" for jobs that had already
    exited. Reading /proc and excluding this process and its ancestors is exact,
    and matching on the interpreter path inside our own envs/ directory means no
    process belonging to the other eight users of this machine can be selected.
    """
    fragments = [part for part in command.split()
                 if part.endswith(".py") or part.startswith("SHARED_")]
    if not fragments:
        return []
    protect = _ancestors(os.getpid())
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in protect:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        if str(ROOT / "envs") not in argv or "pgrep" in argv or argv.startswith("bash -c"):
            continue
        if all(fragment in argv for fragment in fragments[:2]):
            found.append(pid)
    return found


def dump_stacks(job: Path) -> str:
    """Ask a stalled job to print where it is, via the SIGUSR1 hook.

    Every matching process is signalled, not just the runner: the hang may be in a
    pool worker or a generated-code sandbox. The result is reported from whether the
    dump file actually grew, so "signal sent" is never mistaken for "stacks
    captured" -- a process started before shared/pyhooks was on its PYTHONPATH has
    no handler and will ignore the signal entirely.
    """
    # Newest .cmd first: a job in its test phase must be matched on the test
    # command, not on the search command it finished with.
    candidates = [job / f"{p}.cmd" for p in ("test", "search")
                  if (job / f"{p}.cmd").exists()]
    command_file = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    if command_file is None:
        return "no .cmd file; cannot identify the process"
    pids = find_job_pids(command_file.read_text(encoding="utf-8").strip())
    if not pids:
        return "no live process: the job died without writing a status file"

    dump = job / "stacks.txt"
    before = dump.stat().st_size if dump.exists() else 0
    for pid in pids[:8]:
        try:
            os.kill(pid, signal.SIGUSR1)
        except (ProcessLookupError, PermissionError) as exc:
            return f"could not signal {pid}: {type(exc).__name__}"
    time.sleep(3)
    after = dump.stat().st_size if dump.exists() else 0
    if after > before:
        return f"stacks captured from {len(pids[:8])} pid(s) in {dump.name}"
    return (f"signalled {len(pids[:8])} pid(s) but {dump.name} did not grow: the "
            f"process predates the SIGUSR1 hook, or is blocked in a way that "
            f"cannot run a handler")


def job_health(last_seen: dict[str, float] | None = None) -> dict:
    """Per-job status, the counters that mean samples are being lost, and stalls."""
    jobs: dict[str, dict] = {}
    last_seen = last_seen or {}
    now = time.time()
    for search_log in sorted(RUNS.glob("*/*/repeat*/search.log")):
        job = search_log.parent
        name = f"{job.parents[1].name}/{job.parents[0].name}"
        status_file = job / "status"
        status = (status_file.read_text(encoding="utf-8").strip()
                  if status_file.exists() else "running")

        # The phase this job is CURRENTLY in, not the first one it ran.
        #
        # Looking only at search.log made every job that reached its test phase
        # report as stalled 45 minutes later, by construction: MaAS and DAAO write
        # to test.log and send on test/<method>/<dataset>, so both the log-quiet
        # and the wire-quiet signals were reading a phase that had legitimately
        # finished. Measured on daao/drop, which was flagged while its test phase
        # was at 957/1000 with 6,802 requests on the wire.
        logs = [p for p in (job / "search.log", job / "test.log") if p.exists()]
        log_path = max(logs, key=lambda p: p.stat().st_mtime)
        phase = log_path.stem
        text = log_path.read_text(encoding="utf-8", errors="replace")

        quiet_log = (now - log_path.stat().st_mtime) / 60.0
        namespace_file = job / f"{phase}.namespace"
        namespace = (namespace_file.read_text(encoding="utf-8").strip()
                     if namespace_file.exists() else "")
        seen = last_seen.get(namespace)
        quiet_wire = (now - seen) / 60.0 if seen else None

        stalled = (status == "running" and quiet_log > STALL_MINUTES
                   and (quiet_wire is None or quiet_wire > STALL_MINUTES))
        jobs[name] = {
            "status": status,
            # A sample that raises is not scored wrong, it is discarded -- the
            # failure mode that silently halved MaAS/DAAO on MMLU-Pro.
            "sample_failed": text.count("sample failed"),
            "fallbacks": text.count("falling back to the first solution"),
            "log_lines": text.count("\n"),
            "quiet_log_min": round(quiet_log, 1),
            "quiet_wire_min": round(quiet_wire, 1) if quiet_wire is not None else None,
            "namespace": namespace,
            "phase": phase,
            "stalled": stalled,
        }
        # A sentinel, not the dump file itself: sitecustomize opens stacks.txt
        # at job start, so it always exists and the old condition never dumped.
        if stalled and not (job / ".stackdump_requested").exists():
            jobs[name]["stack_dump"] = dump_stacks(job)
            (job / ".stackdump_requested").write_text("")
    return jobs


def check() -> dict:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    stats = proxy_stats()
    action = None
    if stats is None:
        action = restart_proxy()
        stats = proxy_stats()

    totals = (stats or {}).get("totals", {})
    requests = totals.get("requests", 0) or 0
    usage = shutil.disk_usage("/home/users")
    free_gb = usage.free / 1024 ** 3

    driver_alive = bool(shell("pgrep -f 'sweep[.]py --repeats' | head -1"))
    # Ports that actually answer, not processes that exist. `pgrep vllm serve`
    # counts the launcher and its workers, so one dead instance still leaves three
    # matching lines -- the count would stay above the threshold and the alert
    # would never fire. An instance that cannot answer /v1/models is down whatever
    # its processes are doing.
    vllm_alive = sum(
        1 for port in (os.getenv("VLLM_PORTS", "8001,8002")).split(",")
        if shell(f"curl -s -m 5 -o /dev/null -w '%{{http_code}}' "
                 f"http://127.0.0.1:{port.strip()}/v1/models") == "200")
    last_seen = namespace_last_seen()
    jobs = job_health(last_seen)
    findings = contamination()
    miscored = miscore_scan()

    record = {
        "time": now,
        "driver_alive": driver_alive,
        "vllm_instances": vllm_alive,
        "proxy_action": action,
        "proxy_requests": requests,
        "proxy_failure_rate": round((totals.get("failures", 0) or 0) / max(requests, 1), 4),
        "free_gb": round(free_gb, 1),
        "jobs_done": sum(1 for j in jobs.values() if j["status"] == "ok"),
        "jobs_failed": sum(1 for j in jobs.values() if j["status"].startswith("failed")),
        "jobs_running": sum(1 for j in jobs.values() if j["status"] == "running"),
        "samples_discarded": sum(j["sample_failed"] for j in jobs.values()),
        "ensemble_fallbacks": sum(j["fallbacks"] for j in jobs.values()),
        "jobs_stalled": sum(1 for j in jobs.values() if j["stalled"]),
        "contaminated_cells": len(findings),
    }

    concerns = []
    if not driver_alive:
        concerns.append("the sweep driver is gone: no new jobs will start")
    if vllm_alive < 2:
        concerns.append(f"only {vllm_alive} vLLM instance(s) serving; throughput is halved "
                        f"and a restart on a shared GPU needs a human")
    if action:
        concerns.append(f"proxy was unreachable and was restarted ({action})")
    if free_gb < DISK_FLOOR_GB:
        concerns.append(f"{free_gb:.0f} GB free on a disk nine people share")
    if record["jobs_failed"]:
        failed = [n for n, j in jobs.items() if j["status"].startswith("failed")]
        concerns.append(f"{len(failed)} job(s) failed: {', '.join(sorted(failed)[:6])}")
    if record["samples_discarded"]:
        worst = sorted(jobs.items(), key=lambda kv: -kv[1]["sample_failed"])[:3]
        concerns.append("samples are being discarded: "
                        + ", ".join(f"{n} x{j['sample_failed']}" for n, j in worst if j["sample_failed"]))
    if findings:
        detail = ", ".join(f"{f['method']}/{f['dataset']} carries {f['wording']} "
                           f"wording in {f['share']:.0%} of prompts"
                           for f in findings[:4])
        concerns.append(f"prompt wording does not fit the dataset: {detail}")
    if miscored:
        concerns.append("GRADING SUSPECTS in fresh records: " + "; ".join(miscored[:4]))
    stalled = [(n, j) for n, j in jobs.items() if j["stalled"]]
    if stalled:
        detail = ", ".join(
            f"{n} (log quiet {j['quiet_log_min']:.0f}min"
            + (f", wire quiet {j['quiet_wire_min']:.0f}min" if j['quiet_wire_min'] is not None
               else ", namespace never seen")
            + (f"; {j['stack_dump']}" if j.get("stack_dump") else "") + ")"
            for n, j in stalled[:4])
        concerns.append(f"{len(stalled)} job(s) appear stalled: {detail}")
    if record["proxy_failure_rate"] > 0.05:
        concerns.append(f"proxy failure rate {record['proxy_failure_rate']:.1%}")
    record["concerns"] = concerns

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [f"watchdog {now}",
             f"  driver={'alive' if driver_alive else 'GONE'} vllm={vllm_alive} "
             f"proxy_requests={requests:,} failure_rate={record['proxy_failure_rate']:.2%}",
             f"  jobs: {record['jobs_done']} done / {record['jobs_running']} running / "
             f"{record['jobs_failed']} failed / {record['jobs_stalled']} stalled   "
             f"discarded_samples={record['samples_discarded']}",
             f"  free={free_gb:.0f}GB  contaminated_cells={record['contaminated_cells']}"]
    lines += [f"  CONCERN: {c}" for c in concerns] or ["  no concerns"]
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - a watchdog that dies is worse than a noisy one
            print(f"[watchdog] check failed: {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
