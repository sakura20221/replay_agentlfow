#!/usr/bin/env python3
"""Stage gate 0: everything that can be verified WITHOUT spending GPU time.

Run before any sweep, and again after any re-seed. Every restart this project has
suffered traces back to a condition this file now checks: an installer patch that
did not take, a scorer that mis-grades its own gold answers, an environment where
sympy silently cannot parse LaTeX, a workspace still carrying a previous run's
rounds, a serving window smaller than the proxy assumes. Each check prints one
line; any failure makes the exit code non-zero, and the pipeline refuses to
advance past a non-zero gate.

    envs/maas/bin/python preflight.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
failures: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{'  -- ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def run(label: str, command: list[str], must_contain: str | None = None,
        timeout: int = 600) -> None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        report(False, label, "timed out")
        return
    except OSError as exc:
        report(False, label, f"cannot start command: {exc}")
        return
    output = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and (must_contain is None or must_contain in output)
    detail = "" if ok else output.strip().splitlines()[-1][:140] if output.strip() else "no output"
    report(ok, label, detail)


def main() -> None:
    print("### preflight: static gates, no GPU spend ###\n")

    # 0. Frozen inputs and runtime prompts. These checks need no model server and
    # catch identity/boundary errors before any benchmark process is launched.
    run("frozen split identity, hashes and boundaries",
        [sys.executable, "scripts/verify_data.py"])
    run("train-then-eval concatenations",
        [sys.executable, "make_train_then_eval.py", "--check"])
    run("runtime prompt contamination (all methods/datasets)",
        [sys.executable, "audits/scan_prompt_contamination.py"])
    run("FlowBank UID aggregation regression",
        [sys.executable, "audits/test_flowbank_uid_aggregation.py"])
    run("complete sweep matrix and boundary regression",
        [sys.executable, "audits/test_sweep_matrix.py"])
    run("protocol isolation and resume regression",
        [sys.executable, "audits/test_protocol_isolation.py"])
    run("collector strictness regression",
        [sys.executable, "audits/test_collector_strictness.py"])

    # 1. Serving side: both vLLM instances up, at the window the proxy assumes.
    for port in (8001, 8002):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                        timeout=10) as response:
                data = json.loads(response.read())
            window = data["data"][0].get("max_model_len")
            report(window == 40960, f"vLLM :{port} serving max_model_len=40960",
                   f"got {window}")
        except Exception as exc:  # noqa: BLE001
            report(False, f"vLLM :{port} reachable", f"{type(exc).__name__}")
    try:
        with urllib.request.urlopen("http://127.0.0.1:18080/stats", timeout=10) as response:
            json.loads(response.read())
        report(True, "proxy reachable")
    except Exception as exc:  # noqa: BLE001
        report(False, "proxy reachable", f"{type(exc).__name__}")

    # 2. Every installer's own assertions.
    for env, shim in (("tools", "shims/maas_family/install.py"),
                      ("maas", "shims/aflow/install.py"),
                      ("maas", "shims/diverseflow/install.py"),
                      ("gdesigner", "shims/gdesigner_family/install.py"),
                      ("pyg", "shims/masrouter/install.py")):
        executable = ROOT / f"envs/{env}/bin/python"
        try:
            proc = subprocess.run([str(executable), shim, "--check"],
                                  capture_output=True, text=True, cwd=ROOT, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report(False, f"{shim} --check", f"cannot run {executable}: {exc}")
            continue
        bad = [l for l in (proc.stdout or "").splitlines() if "[FAIL" in l]
        report(proc.returncode == 0 and not bad, f"{shim} --check",
               bad[0].strip() if bad else "")

    # 3. The scorer must give every frozen gold answer full marks.
    python = str(ROOT / "envs/maas/bin/python")
    run("MBPP public-test identity lookup",
        [python, "audits/test_mbpp_public_lookup.py"])
    for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
        run(f"gold roundtrip: {dataset}",
            [python, "audits/test_gold_roundtrip.py", "--dataset", dataset], "[OK]")

    # 4. LaTeX equivalence must work in every env that ever grades maths.
    for env in ("maas", "gdesigner", "pyg"):
        run(f"sympy parses LaTeX in envs/{env}",
            [str(ROOT / f"envs/{env}/bin/python"), "-c",
             "from sympy.parsing.latex import parse_latex; parse_latex(r'\\frac{1}{2}')"])

    # 5. Proxy protocol and namespace plumbing.
    run("proxy sampling protocol (20 assertions)",
        [str(ROOT / "envs/tools/bin/python"), "audits/test_proxy_protocol.py"])
    for env, repo in (("gdesigner", "third_party/gdesigner"),
                      ("maas", "third_party/maas"),
                      ("maas", "third_party/aflow")):
        run(f"namespace override reaches {repo.split('/')[-1]}",
            [str(ROOT / f"envs/{env}/bin/python"), "audits/test_namespace.py",
             "--repo", repo])

    # 6. Workspaces must hold ONLY the seed. A leftover round is how a "search"
    #    finished in 0.9 minutes and reported a previous protocol's workflow.
    for pattern, label in (
            ("third_party/aflow/workspace/SHARED_*/workflows", "aflow"),
            ("third_party/flowbank/DiverseFlow/workspace/SHARED_*/workflows", "flowbank")):
        stale = []
        for workflows in ROOT.glob(pattern):
            rounds = sorted(p.name for p in workflows.glob("round_*") if p.is_dir())
            if rounds != ["round_1"]:
                stale.append(f"{workflows.parent.name}:{rounds}")
        report(not stale, f"{label} workspaces hold only the seed round",
               "; ".join(stale[:3]))
    leftover_results = [str(p.relative_to(ROOT))
                        for p in ROOT.glob("third_party/*/result/*/")
                        if p.is_dir() and any(p.iterdir())]
    report(not leftover_results, "no leftover per-item result directories",
           "; ".join(leftover_results[:3]))

    print(f"\n### preflight: {len(failures)} failure(s) ###")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
