#!/usr/bin/env python3
"""The whole rerun as one gated sequence. Errors surface in minutes, not hours.

Every restart so far followed the same shape: launch the full sweep, discover an
input-side defect hours in, archive, start over. The fix is ordering, not effort --
all of those defects were detectable cheaply, before any GPU hour was spent, by
exercising the real code paths at toy scale and auditing what actually flowed.

    stage 0  clean+seed   archive every artefact path, re-run all installers
    stage 1  preflight    static gates (installers, scorer regression, envs,
                          window, workspace cleanliness) -- minutes, no GPU
    stage 2  smoke        every (method,dataset) cell end-to-end on 6 items
    stage 3  smoke audit  contamination scan + collect must reach every cell +
                          correct-but-zero on the smoke's own per-item files
    stage 4  clean+seed   remove smoke artefacts, verify clean again
    stage 5  full sweep   drop+math+mmlu_pro wave, then mbpp wave (resumable)
    stage 6  stop         finishers (aflow_test / flowbank_pipeline) stay manual:
                          aflow_test must re-grade rounds before choosing one

A stage that fails stops the pipeline with its log; a re-run resumes at the first
gate that has not passed (gate files under logs/pipeline/). Grading-side changes
never re-enter this pipeline: collect.py re-grades stored predictions, so a scorer
fix is applied by re-collecting, not by re-running.

    tmux: envs/tools/bin/python pipeline.py --runs runs_v5 --tag v5
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATES = ROOT / "logs" / "pipeline"

INSTALLERS = (("tools", "shims/maas_family/install.py"),
              ("maas", "shims/aflow/install.py"),
              ("maas", "shims/diverseflow/install.py"),
              ("gdesigner", "shims/gdesigner_family/install.py"),
              ("pyg", "shims/masrouter/install.py"))


def log(message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sh(command: str, timeout: int | None = None, env: dict | None = None) -> tuple[int, str]:
    import os

    merged = dict(os.environ)
    if env:
        merged.update(env)
    proc = subprocess.run(command, shell=True, cwd=ROOT, env=merged,
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def gate(name: str):
    """Skip a stage whose gate already passed, so re-runs resume mid-pipeline."""
    marker = GATES / f"{name}.passed"

    def decorator(fn):
        def wrapped(*args, **kwargs):
            if marker.exists():
                log(f"stage {name}: already passed ({marker.read_text().strip()}), skipping")
                return
            log(f"stage {name}: start")
            fn(*args, **kwargs)
            marker.write_text(datetime.datetime.now().isoformat() + "\n")
            log(f"stage {name}: PASSED")
        return wrapped
    return decorator


def fail(stage: str, detail: str) -> None:
    log(f"stage {stage}: FAILED -- {detail}")
    sys.exit(1)


def clean_and_seed(label: str) -> None:
    code, out = sh(f"envs/tools/bin/python separate_runs.py --label {label} --apply",
                   timeout=1200)
    # "nothing to move" is a legitimate clean state, not a failure.
    if code != 0 and "live job" in out:
        fail(label, "live jobs detected; stop them first")
    for env, shim in INSTALLERS:
        code, out = sh(f"envs/{env}/bin/python {shim}", timeout=900)
        if code != 0 or "[FAIL" in out:
            fail(label, f"{shim}: " + next((l for l in out.splitlines() if "[FAIL" in l),
                                           out.strip().splitlines()[-1] if out.strip() else "?"))


def preflight(stage: str) -> None:
    code, out = sh("envs/maas/bin/python preflight.py", timeout=1800)
    (GATES / f"{stage}.report").write_text(out)
    if code != 0:
        fail(stage, "see logs/pipeline/" + stage + ".report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs_v5")
    parser.add_argument("--tag", default="v5")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--smoke-timeout", type=int, default=3600)
    args = parser.parse_args()
    GATES.mkdir(parents=True, exist_ok=True)
    runs = str(ROOT / args.runs)
    smoke_runs = str(ROOT / (args.runs + "_smoke"))

    @gate("0_clean")
    def stage0() -> None:
        clean_and_seed("pipeline_pre_clean")

    @gate("1_preflight")
    def stage1() -> None:
        preflight("1_preflight")

    @gate("2_smoke")
    def stage2() -> None:
        code, out = sh(
            f"SWEEP_RUNS={smoke_runs} SWEEP_TAG=smoke "
            f"envs/tools/bin/python sweep.py --smoke --repeats 1 --jobs {args.jobs} "
            f"--timeout {args.smoke_timeout} --datasets drop math mmlu_pro mbpp",
            timeout=args.smoke_timeout * 4)
        (GATES / "2_smoke.report").write_text(out)
        failed = [l for l in out.splitlines() if "FAILED" in l]
        if code != 0 or failed:
            fail("2_smoke", f"{len(failed)} smoke job(s) failed; first: "
                            f"{failed[0] if failed else out.strip().splitlines()[-1]}")

    @gate("3_smoke_audit")
    def stage3() -> None:
        problems = []
        # Prompt fit, on the smoke's own live traffic.
        code, out = sh("envs/tools/bin/python audits/live_contamination.py --json",
                       timeout=600)
        findings = json.loads(out).get("findings", []) if out.startswith("{") else None
        if findings is None:
            problems.append("contamination scan unreadable")
        elif findings:
            problems.append(f"contaminated: {findings[:2]}")
        # Grading health on the smoke's own per-item files.
        code, out = sh("envs/maas/bin/python audits/correct_but_zero.py "
                       "--tag smoke --datasets drop math mmlu_pro mbpp", timeout=1200)
        (GATES / "3_smoke_audit.report").write_text(out)
        if code != 0:
            problems.append("correct_but_zero crashed")
        # Collection must reach every smoke cell (this is the gate that catches a
        # tag mismatch or a storage-format change before the real run).
        code, out = sh(f"SWEEP_RUNS={smoke_runs} envs/maas/bin/python collect.py",
                       timeout=900)
        (GATES / "3_smoke_collect.report").write_text(out)
        missing = [l for l in out.splitlines()
                   if "no result file" in l or "none carried" in l]
        if missing:
            problems.append(f"collect cannot reach {len(missing)} cell(s): "
                            f"{missing[0].split()[0]}...")
        if problems:
            fail("3_smoke_audit", "; ".join(problems))

    @gate("4_clean_again")
    def stage4() -> None:
        clean_and_seed("pipeline_post_smoke")
        preflight("4_clean_again")

    @gate("5_full_sweep")
    def stage5() -> None:
        # The watchdog restarts pointed at the real runs directory.
        sh("tmux kill-session -t sml_watch 2>/dev/null")
        sh(f"tmux new-session -d -s sml_watch -c {ROOT} "
           f"'SWEEP_RUNS={runs} envs/tools/bin/python watchdog.py --interval 600 "
           f">> logs/watchdog_pipeline.log 2>&1'")
        for wave, datasets in (("wave1", "drop math mmlu_pro"), ("wave2", "mbpp")):
            log(f"stage 5 {wave}: {datasets}")
            code, out = sh(
                f"SWEEP_RUNS={runs} SWEEP_TAG={args.tag} "
                f"envs/tools/bin/python sweep.py --repeats 1 --jobs {args.jobs} "
                f"--timeout 43200 --datasets {datasets}",
                timeout=None)
            (GATES / f"5_{wave}.report").write_text(out[-100_000:])
            failed = [l for l in out.splitlines() if "FAILED" in l]
            if code != 0 or failed:
                fail("5_full_sweep", f"{wave}: {len(failed)} job(s) failed; the sweep "
                                     f"is resumable -- fix and re-run the pipeline")

    stage0()
    stage1()
    stage2()
    stage3()
    stage4()
    stage5()
    log("all gates passed. Finishers are manual on purpose:")
    log("  1. aflow_test.py -- re-grade rounds (audits/regrade_rounds.py) BEFORE it picks one")
    log("  2. flowbank_pipeline.py stages 2a-3e")
    log(f"  3. SWEEP_RUNS={runs} envs/maas/bin/python collect.py")


if __name__ == "__main__":
    main()
