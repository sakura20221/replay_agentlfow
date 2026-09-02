#!/usr/bin/env python3
"""Run the bake-off matrix: method x dataset x repeat, search phase then test phase.

Design notes that matter for reading the results:

* A job is (method, dataset, repeat). Repeats are *independent process runs*, not
  seeded variants: most of these repos do not expose a seed, and patching one in
  would be a method modification. Run-to-run spread therefore comes from whatever
  each repo leaves unseeded plus vLLM's continuous batching, which is not
  bitwise deterministic even at temperature 0 (measured: +-1.2 points at n=256).
  That spread is the reason the table needs mean +- std rather than single runs.

* Search and test are separate commands per method because every repo splits them
  differently. The test command is the one whose number goes in the table; the
  search command is what produces the artifact it consumes.

* Resumable: a job whose `status` file says `ok` is skipped, so an interrupted
  sweep can be restarted without redoing finished work. Delete the status file to
  force a rerun.

* Concurrency is capped because all jobs share two vLLM instances behind the
  proxy. Raising it past the point where the instances saturate only lengthens
  every job's wall clock; it does not add throughput.

    python sweep.py --list
    python sweep.py --methods maas --datasets math --repeats 1
    python sweep.py --phase search --jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# SWEEP_RUNS lets a new protocol write to its own tree instead of on top of an
# earlier one: scores produced under different prompts are not comparable, and
# the cheapest way to guarantee they never share a table is separate directories.
RUNS = Path(os.getenv("SWEEP_RUNS", str(ROOT / "runs")))
# The same tag namespaces the per-item result files the G-Designer family writes
# *inside its own repo*. Their runner appends, so without this a rerun would add
# new records to the file holding the previous protocol's records -- and that file
# is exactly what the final accuracy is recomputed from.
RUN_TAG = os.getenv("SWEEP_TAG", RUNS.name)
# ...and recorded next to the runs, because collect.py derives the same default and
# the two disagree the moment SWEEP_TAG is passed to one and not the other. That
# mismatch is silent and reads as "no result file yet" -- i.e. exactly like a job
# that produced nothing, which is the wrong conclusion about a job that finished.
try:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "TAG").write_text(RUN_TAG + "\n", encoding="utf-8")
except OSError:
    pass
sys.path.insert(0, str(ROOT / "shared"))
from bench import data_fingerprint, protocol_fingerprint  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_proxy import sampling_protocol  # noqa: E402 - the proxy owns the sampling rules

# Where the proxy listens. Kept here rather than repeated in seven config files:
# the per-job namespace is appended to it below.
PROXY_ROOT = os.getenv("SWEEP_PROXY_ROOT", "http://127.0.0.1:18080")
ENVS = ROOT / "envs"

DATASETS = ("math", "amc", "mbpp", "drop", "mmlu_pro")

# dataset -> the key each repo family uses for it
SHARED_KEY = {d: f"SHARED_{d.upper().replace('_', '')}" for d in DATASETS}

# How many optimisation rounds / epochs each searching method gets. These are the
# authors' own defaults where they state one; FlowBank's 30 comes from its README
# and its diversity phase does not even begin until round 7.
ROUNDS = {"aflow": 20, "flowbank": 30}

# Relative cost, used only for scheduling order. Longest-first packing: with 12
# slots and 35 jobs the wall clock is dominated by whatever starts last, and in
# alphabetical order that was flowbank -- 30 optimisation rounds per job, five of
# them, left to run nearly alone after everything else had drained. Numbers are
# round budgets where a method has one and measured relative length otherwise
# (card/mbpp search: 37min).
COST = {"flowbank": 30, "aflow": 20, "maas": 8, "daao": 8,
        "card": 6, "gdesigner": 6, "masrouter": 3,
        "card_authordefault": 1, "gdesigner_authordefault": 1}

# --eval_batch_size 32 on the G-Designer family: batch_size stays at the author's 4
# for the training batches (it is the gradient batch), but after the switch to
# evaluation the optimiser step is guarded off, so there the batch is only a
# concurrency knob. Measured at batch 4 these two dispatch 4 requests at a time
# against everyone else's 30-50, which put one mmlu_pro job on a 34-hour path.

# Bigger evaluation splits take longer, so they go first within a method.
DATASET_COST = {"mmlu_pro": 1120, "drop": 1000, "math": 486, "mbpp": 500, "amc": 648}

# Exact search boundaries and corresponding numbers of gradient batches.
# math switched to the AFlow/FlowBank official Level-5 split (119 validate + 486
# test) on 2026-08-24, replacing MATH-500: the published FlowBank/AFlow numbers
# are on that split, and MATH-500's mixed difficulty put Qwen3-8B near its
# ceiling. The runner accepts a smaller final training batch, so no item is cut.
TRAIN_ITEMS = {"math": 119, "amc": 165, "mbpp": 256, "drop": 256, "mmlu_pro": 252}
TRAIN_BATCHES = {"math": 30, "amc": 42, "mbpp": 64, "drop": 64, "mmlu_pro": 63}


# Cells that are deliberately not run, with the reason. Declared here rather than
# left out of a command line, so the gap is visible in the matrix listing and in
# the log instead of looking like a job that failed.
#
# The overall average must then be taken over the datasets every method ran; a
# mean over different dataset sets per method is not a comparison.
# Empty: every method runs every dataset.
#
# masrouter/drop was briefly excluded on the grounds that reading comprehension is
# none of MasRouter's three task types. That was the wrong call. The authors' own
# run_mmlu.py routes MMLU to Commonsense, which is their question-answering pool,
# and DROP belongs there on the same reasoning -- so the substitution is the one
# they already make, not a new one. What DROP genuinely needed was wording: three
# of the seven Commonsense roles state an answer shape ("choose the correct
# answer", "a complex math problem") that is wrong for a span, and the final
# decision node was being handed MATH's boxed-answer prompt. Both are adapted in
# shims/masrouter/install.py, with the three task types left untouched so no other
# dataset's routing moves.
EXCLUDED: dict[tuple[str, str], str] = {}


# --smoke appends these AFTER each method's own flags; argparse takes the last
# occurrence, so the template stays untouched and the override is visible in the
# job's .cmd file. maas/daao need no flags: their item cap (SHIM_SMOKE_N, honoured
# by SharedBenchmark.load_data) is the whole story, and --round is already 1.
# masrouter's batch_size must shrink below the item cap or int(6/16)=0 batches
# would skip the loops entirely and the smoke would validate nothing.
SMOKE_APPEND = {
    "gdesigner": " --num_iterations 1 --train_items 4 --search_items 4",
    "card": " --num_iterations 1 --train_items 4 --search_items 4",
    "gdesigner_authordefault": " --num_iterations 1 --train_items 4 --search_items 4",
    "card_authordefault": " --num_iterations 1 --train_items 4 --search_items 4",
    "masrouter": " --batch_size 2 --max_batches 1 --epochs 1",
    "aflow": " --max_rounds 1",
    "flowbank": " --max_rounds 1 --check_convergence false",
}
SMOKE_ITEMS = 6
# Set from --smoke in main(); run_job runs in worker threads and must not reach
# into main()'s locals.
SMOKE_MODE = False


def _py(env: str) -> str:
    return str(ENVS / env / "bin" / "python")


# method -> (cwd, env, search command, test command)
#
# {key} is the SHARED_* dataset key, {ds} the lowercase name, {rounds} the round
# budget, {out} the job's output directory.
METHODS: dict[str, dict] = {
    "maas": {
        "cwd": "third_party/maas",
        "env": "maas",
        "search": "{py} examples/maas/optimize.py --dataset {key} --sample 4 --round 1"
                  " --batch_size 4 --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        # MaAS declares --is_test as type=bool, so it needs a value; DAAO declares
        # the same flag as action="store_true", where a value is a parse error.
        # Note MaAS's form also means --is_test False would evaluate to True.
        "test": "{py} examples/maas/optimize.py --dataset {key} --sample 4 --round 1"
                " --is_test True --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        "pythonpath": ".",
        # test() does `if os.path.exists(controller_path): load` -- with no else.
        # A missing checkpoint means it evaluates an *untrained* controller and
        # still prints a score, so the file is asserted rather than hoped for.
        "test_requires": "maas/ext/maas/scripts/optimized/{key}/train/round_1/"
                         "{key}_controller_sample4.pth",
    },
    "daao": {
        "cwd": "third_party/daao",
        "env": "maas",
        "search": "{py} examples/maas/optimize.py --dataset {key} --sample 4 --round 1"
                  " --batch_size 4 --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        "test": "{py} examples/maas/optimize.py --dataset {key} --sample 4 --round 1"
                " --is_test --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        "pythonpath": ".",
        "test_requires": "daao/ext/maas/scripts/optimized/{key}/train/round_1/"
                         "{key}_controller_sample4.pth",
    },
    # One process, one file: train on the search split, then evaluate on held-out
    # data with the topology just learned. --num_iterations is the search split's
    # batch count, so every method sees the complete frozen search split for that
    # dataset. The earlier
    # two-process split started the test run from an untrained GCN and ran
    # backward()/step() on the evaluation data.
    "gdesigner": {
        "cwd": "third_party/gdesigner",
        # Per-job result file: the default name is {domain}_{llm}_{timestamp},
        # and two jobs of the same domain starting in the same second would append
        # their per-item records to one file -- the file the final accuracy is
        # recomputed from.
        "result_file": "result/{tag}/gdesigner_{ds}_r{repeat}.json",
        "env": "gdesigner",
        "search": "{py} experiments/run_shared.py"
                  " --dataset_json datasets/shared/{ds}_train_then_eval.jsonl"
                  " --domain {ds} --llm_name qwen3-8b --batch_size 4"
                  " --num_iterations {train_batches} --train_items {train_items}"
                  " --search_items {train_items} --num_rounds 1 --optimized_spatial --eval_batch_size 32",
        "test": None,
    },
    "card": {
        "cwd": "third_party/card",
        # Per-job result file: the default name is {domain}_{llm}_{timestamp},
        # and two jobs of the same domain starting in the same second would append
        # their per-item records to one file -- the file the final accuracy is
        # recomputed from.
        "result_file": "result/{tag}/card_{ds}_r{repeat}.json",
        "env": "gdesigner",
        "search": "{py} experiments/run_shared.py"
                  " --dataset_json datasets/shared/{ds}_train_then_eval.jsonl"
                  " --domain {ds} --llm_name qwen3-8b --mode FullConnected --batch_size 4"
                  " --num_iterations {train_batches} --train_items {train_items}"
                  " --search_items {train_items} --num_rounds 1 --optimized_spatial --eval_batch_size 32",
        "test": None,
    },
    # Author-default budget, kept as a control row: num_iterations=10 is 40 items
    # rather than the complete dataset-specific search split. Reported separately,
    # never mixed into the seven-method primary table --
    # 64 gradient steps at lr 0.1 could destabilise the topology learner, and if
    # unifying the training set hurts these two, that has to be visible.
    "gdesigner_authordefault": {
        "cwd": "third_party/gdesigner",
        # Per-job result file: the default name is {domain}_{llm}_{timestamp},
        # and two jobs of the same domain starting in the same second would append
        # their per-item records to one file -- the file the final accuracy is
        # recomputed from.
        "result_file": "result/{tag}/gdesigner_authordefault_{ds}_r{repeat}.json",
        "env": "gdesigner",
        "search": "{py} experiments/run_shared.py"
                  " --dataset_json datasets/shared/{ds}_train_then_eval.jsonl"
                  " --domain {ds} --llm_name qwen3-8b --batch_size 4"
                  " --num_iterations 10 --train_items 40 --search_items {train_items}"
                  " --num_rounds 1 --optimized_spatial --eval_batch_size 32",
        "test": None,
    },
    "card_authordefault": {
        "cwd": "third_party/card",
        # Per-job result file: the default name is {domain}_{llm}_{timestamp},
        # and two jobs of the same domain starting in the same second would append
        # their per-item records to one file -- the file the final accuracy is
        # recomputed from.
        "result_file": "result/{tag}/card_authordefault_{ds}_r{repeat}.json",
        "env": "gdesigner",
        "search": "{py} experiments/run_shared.py"
                  " --dataset_json datasets/shared/{ds}_train_then_eval.jsonl"
                  " --domain {ds} --llm_name qwen3-8b --mode FullConnected --batch_size 4"
                  " --num_iterations 10 --train_items 40 --search_items {train_items}"
                  " --num_rounds 1 --optimized_spatial --eval_batch_size 32",
        "test": None,
    },
    "masrouter": {
        "cwd": "third_party/masrouter",
        "env": "pyg",
        # One process trains on the search split and then evaluates on the test
        # split, so there is no separate test command.
        "search": "{py} Experiments/run_shared.py --shared_dataset {ds} --domain {ds}"
                  # batch_size=16 and epochs=5 are both the author defaults
                  # (run_math.py). The 4 and 2 used earlier were my own inventions:
                  # they gave MasRouter a different gradient variance and less data
                  # than its paper setting. Five epochs over the frozen search pool means
                  # it sees every training question, like every other method here.
                  " --batch_size 16 --epochs 5 --num_rounds 1",
        "test": None,
    },
    "aflow": {
        "cwd": "third_party/aflow",
        "env": "maas",
        "search": "{py} run.py --dataset {key} --sample 4 --max_rounds {rounds}"
                  " --validation_rounds 1 --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        # Upstream run.py only searches on the validation split. The local driver
        # materialises the highest-scoring searched round and invokes the authors'
        # otherwise-unreachable Test path on the held-out split.
        "test": "{py} ../../aflow_test.py --dataset {key} --sample 4"
                " --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        "test_requires": "workspace/{key}/workflows/results.json",
    },
    "flowbank": {
        "cwd": "third_party/flowbank/DiverseFlow",
        "env": "maas",
        "search": "{py} run.py --dataset {key} --sample 4 --max_rounds {rounds}"
                  " --validation_rounds 1 --opt_model_name qwen3-8b --exec_model_name qwen3-8b",
        # Stages 2 and 3 (CuraFlow selection, QueryMatching training) are driven
        # separately: they consume the whole round pool at once rather than one
        # (dataset, repeat) job, so they are not expressible in this matrix.
        "test": None,
    },
}

_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def job_dir(method: str, dataset: str, repeat: int) -> Path:
    return RUNS / method / dataset / f"repeat{repeat}"


def build(method: str, dataset: str, phase: str) -> str | None:
    spec = METHODS[method]
    template = spec.get(phase)
    if not template:
        return None
    return template.format(
        py=_py(spec["env"]),
        key=SHARED_KEY[dataset],
        ds=dataset,
        rounds=ROUNDS.get(method, 1),
        train_batches=TRAIN_BATCHES[dataset],
        train_items=TRAIN_ITEMS[dataset],
    )


def current_job_protocol(method: str, dataset: str, repeat: int) -> dict:
    """Every protocol dimension that can change a search artifact or score."""
    return {
        **protocol_fingerprint(),
        "data": data_fingerprint(dataset),
        "sampling": sampling_protocol(),
        "method": method,
        "dataset": dataset,
        "run_tag": RUN_TAG,
        "repeat": repeat,
    }


def protocol_differences(recorded: dict, current: dict) -> list[str]:
    return [key for key, value in current.items() if recorded.get(key) != value]


def run_job(method: str, dataset: str, repeat: int, phases: list[str], timeout: int) -> tuple[str, str]:
    tag = f"{method}/{dataset}/r{repeat}"
    out = job_dir(method, dataset, repeat)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status"

    # Refuse to relabel an already-produced phase with a newer protocol. Without
    # this guard, a resumed job could overwrite protocol.json, skip its old search
    # because search.seconds exists, and test an old artifact under a new stamp.
    protocol_path = out / "protocol.json"
    current_protocol = current_job_protocol(method, dataset, repeat)
    completed_phases = [phase for phase in ("search", "test")
                        if (out / f"{phase}.seconds").exists()]
    if protocol_path.exists():
        try:
            recorded_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            differences = ["missing-or-unreadable-stamp"]
        else:
            differences = protocol_differences(recorded_protocol, current_protocol)
        if differences:
            status_path.write_text("failed protocol mismatch\n", encoding="utf-8")
            prior = f"completed phases {completed_phases}" if completed_phases else "prior run state"
            log(f"{tag}: REFUSING resume; {prior} uses a different protocol "
                f"({', '.join(differences)}). Use a new runs directory and tag.")
            return tag, "failed:protocol-mismatch"
    if status_path.exists() and status_path.read_text(encoding="utf-8").strip() == "ok":
        return tag, "skipped"
    protocol_path.write_text(
        json.dumps({**current_protocol,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   indent=2) + "\n",
        encoding="utf-8")

    spec = METHODS[method]
    env = dict(os.environ)
    # shared/pyhooks goes first so its sitecustomize is imported at startup: that
    # is what registers SIGUSR1 -> stack dump in every job, which is the only way
    # to see inside a hung job on this host (ptrace_scope=1 blocks py-spy/gdb from
    # attaching to a process that is not their child).
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "shared" / "pyhooks"),
                          spec.get("pythonpath", "") or env.get("PYTHONPATH", ""))
        if part)
    env["SHIM_STACKDUMP"] = str(out / "stacks.txt")
    # One core's worth of BLAS threads per job, not all 128.
    #
    # The client-side models are tiny (a topology GCN, a difficulty VAE, a MiniLM
    # encoder) and run on CPU, but torch/OpenMP size their thread pools from the
    # machine, so 16 jobs each opened ~128-way pools: measured load average 218 on
    # a 128-core box shared with eight other people. The work is bounded by the two
    # vLLM instances -- in-flight requests were already at the 128 measured ceiling
    # -- so those threads bought nothing and slowed everyone, including us.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        env.setdefault(variable, "false" if variable.startswith("TOKENIZERS") else "4")
    # Fatal signals get a traceback too: a segfault inside torch or pyg would
    # otherwise leave only a return code.
    env["PYTHONFAULTHANDLER"] = "1"
    # Pin the client-side torch models -- the topology GCN, the difficulty VAE, the
    # MiniLM sentence encoder -- to a GPU we actually hold. They default to cuda:0,
    # which on this shared machine belongs to someone else and sits at 74 of 80 GB:
    # CARD died with "Tried to allocate 20.00 MiB ... 8.81 MiB is free" while asking
    # for a rounding error's worth of memory. The models are tiny; this is about
    # not landing on a neighbour's card, not about capacity.
    # CPU, not a GPU. These models are tiny -- a topology GCN, a difficulty VAE, a
    # MiniLM encoder -- and every GPU on this shared box is contended: pinning them
    # to cuda:0 landed on a neighbour's full card, and pinning them to our own GPU 7
    # failed too once vLLM's budget there went to 40GB ("Tried to allocate 46.00 MiB
    # ... 26.88 MiB is free"). Keeping them off the GPU entirely removes a failure
    # mode that has now bitten twice and costs nothing measurable.
    env["CUDA_VISIBLE_DEVICES"] = os.getenv("SWEEP_CUDA_DEVICE", "")
    # Per-job instrumentation, so an operator failure or an unusable generated
    # graph is attributable to the job that produced it rather than to a shared
    # file that several jobs append to at once.
    env["AFLOW_GRAPH_FAILURES"] = str(out / "graph_failures.txt")
    env["SHIM_OPERATOR_STATS"] = str(out / "operator_stats.txt")
    env["SHIM_TRACEBACKS"] = "1"
    # Which dataset this job is running. MaAS and DAAO keep one operator template
    # shared by every dataset, so the ScEnsemble label-space fix cannot be decided
    # at install time -- it has to know at runtime whether the answers are letters.
    env["SHIM_DATASET"] = dataset
    # Accounting namespace, per job rather than per method.
    #
    # The proxy takes its namespace from the URL path, and each repo reads that URL
    # from one file in its own checkout -- so every DAAO job, on every dataset, in
    # both phases, reported as "train/daao". Cost per cell and "which cell went
    # quiet" were both unanswerable.
    #
    # The phase segment is filled in per phase below, because search and test are
    # separate commands within this same job.
    #
    # BASE_URL and URL are what the .env-reading repos (G-Designer, CARD,
    # MasRouter) call os.getenv on; python-dotenv does not overwrite an existing
    # environment variable, so exporting them here wins over the file without
    # touching those repos. SHIM_BASE_URL is for the YAML-reading repos, honoured
    # in create_llm_instance by shims/namespace_patch.py.
    # No outbound network on this host. sentence-transformers otherwise probes
    # huggingface.co for an adapter config on every start and burns five retry
    # rounds before falling back to the local copy; the weights are already in
    # shared/models.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # HF_HOME is deliberately left alone. Pointing it at the repo's hf_cache --
    # which holds the Qwen3-8B weights -- made the MiniLM lookup fail outright
    # ("couldn't find them in the cached files"); that copy is incomplete, while
    # the account default cache has a working one.

    for phase in phases:
        command = build(method, dataset, phase)
        if command is None:
            continue
        if spec.get("result_file"):
            path = ROOT / spec["cwd"] / spec["result_file"].format(
                ds=dataset, repeat=repeat, tag=RUN_TAG)
            path.parent.mkdir(parents=True, exist_ok=True)
            command += f" --result_file {path}"
        # Per-phase resume. Without it a failed test phase re-ran its search too --
        # card/math's search alone is 174 minutes, thrown away to retry a 10-minute
        # test command. The marker is written only after a zero exit.
        if (out / f"{phase}.seconds").exists():
            log(f"{tag} {phase}: skipped (already complete)")
            continue

        required = spec.get(f"{phase}_requires")
        if required:
            needed = ROOT / spec["cwd"] / required.format(key=SHARED_KEY[dataset])
            if not needed.exists():
                status_path.write_text(f"failed {phase} missing:{needed.name}\n", encoding="utf-8")
                log(f"{tag} {phase}: FAILED, {phase} needs {needed} and the search "
                    f"phase did not produce it")
                return tag, f"failed:{phase}:missing-artifact"
        if SMOKE_MODE and phase == "search":
            command += SMOKE_APPEND.get(method, "")
        (out / f"{phase}.cmd").write_text(command + "\n", encoding="utf-8")
        # Include the protocol run and repeat. Otherwise an old run, a smoke run,
        # and a formal run all append to the same logical transcript namespace.
        namespace = f"{RUN_TAG}/{phase}/{method}/{dataset}/r{repeat}"
        phase_env = dict(env)
        if SMOKE_MODE:
            phase_env["SHIM_SMOKE_N"] = str(SMOKE_ITEMS)
            # The train/eval boundary of the G-Designer family's concatenated
            # file, in items -- eval smoke items must come from past it so their
            # uids belong to the evaluation split.
            phase_env["SHIM_SMOKE_EVAL_FROM"] = str(TRAIN_ITEMS.get(dataset, 256))
        phase_env["SHIM_NAMESPACE"] = namespace
        phase_env["SHIM_BASE_URL"] = f"{PROXY_ROOT}/{namespace}/v1"
        phase_env["BASE_URL"] = f"{PROXY_ROOT}/{namespace}/v1/chat/completions"
        phase_env["URL"] = f"{PROXY_ROOT}/{namespace}/v1"
        (out / f"{phase}.namespace").write_text(namespace + "\n", encoding="utf-8")
        log(f"{tag} {phase}: start  [{namespace}]")
        started = time.time()
        with (out / f"{phase}.log").open("w", encoding="utf-8") as handle:
            try:
                completed = subprocess.run(
                    shlex.split(command),
                    cwd=ROOT / spec["cwd"],
                    env=phase_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                handle.write(f"\n[sweep] timed out after {timeout}s\n")
                code = -9
        elapsed = time.time() - started
        if code != 0:
            status_path.write_text(f"failed {phase} rc={code}\n", encoding="utf-8")
            log(f"{tag} {phase}: FAILED rc={code} after {elapsed / 60:.1f}min")
            return tag, f"failed:{phase}:rc={code}"
        (out / f"{phase}.seconds").write_text(f"{elapsed:.0f}\n", encoding="utf-8")
        log(f"{tag} {phase}: ok in {elapsed / 60:.1f}min")

    status_path.write_text("ok\n", encoding="utf-8")
    return tag, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=sorted(METHODS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phase", nargs="+", default=["search", "test"],
                        choices=["search", "test"])
    parser.add_argument("--jobs", type=int, default=4,
                        help="concurrent jobs; they share two vLLM instances")
    parser.add_argument("--timeout", type=int, default=12 * 3600,
                        help="per phase, seconds; 6h killed five jobs, and "
                             "aflow/mmlu_pro needed 330min of it")
    parser.add_argument("--list", action="store_true", help="print the matrix and exit")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end run of every cell: 6 items, 1 round/"
                             "iteration/epoch. Validates prompts, proxy, operators, "
                             "per-item files and collection in ~30min before any "
                             "full run burns hours. Appends flag overrides (argparse "
                             "last-wins) and sets SHIM_SMOKE_N for the shim loaders.")
    args = parser.parse_args()
    global SMOKE_MODE
    SMOKE_MODE = args.smoke

    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown methods: {unknown}")

    matrix = [(m, d, r) for m in args.methods for d in args.datasets
              for r in range(1, args.repeats + 1)
              if (m, d) not in EXCLUDED]
    excluded = [(m, d) for m in args.methods for d in args.datasets
                if (m, d) in EXCLUDED]
    if excluded:
        for method, dataset in excluded:
            log(f"excluded: {method}/{dataset} -- {EXCLUDED[(method, dataset)]}")

    # Round-robin across methods, not longest-first.
    #
    # Longest-first minimises total wall clock, but it filled all 12 slots with
    # flowbank and aflow, leaving the five cheaper methods queued for hours -- so a
    # bug in any of them would surface only after those hours. Interleaving gives
    # every method a job from the first minute, which is worth more than optimal
    # packing when per-method breakage has been the dominant failure mode. Within a
    # method the biggest datasets go first, and repeat 1 of everything precedes
    # repeat 2 so an interrupted sweep still yields a complete n=1 table.
    by_method: dict[str, list] = {}
    for job in sorted(matrix, key=lambda j: (j[2], -DATASET_COST.get(j[1], 0))):
        by_method.setdefault(job[0], []).append(job)
    order = sorted(by_method, key=lambda m: -COST.get(m, 1))
    matrix = []
    for index in range(max(len(v) for v in by_method.values())):
        for method in order:
            if index < len(by_method[method]):
                matrix.append(by_method[method][index])

    if args.list:
        for method, dataset, repeat in matrix:
            done = (job_dir(method, dataset, repeat) / "status").exists()
            for phase in args.phase:
                command = build(method, dataset, phase)
                mark = "done" if done else "todo"
                print(f"  [{mark}] {method}/{dataset}/r{repeat} {phase}: "
                      + (command.replace(str(ROOT), ".") if command else "(none)"))
        print(f"\n{len(matrix)} job(s)")
        return

    log(f"{len(matrix)} job(s), {args.jobs} at a time, phases={args.phase}")
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_job, m, d, r, args.phase, args.timeout): (m, d, r)
                   for m, d, r in matrix}
        for future in as_completed(futures):
            tag, outcome = future.result()
            results[tag] = outcome

    summary = RUNS / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    failed = {k: v for k, v in results.items() if v.startswith("failed")}
    log(f"done: {len(results) - len(failed)} ok/skipped, {len(failed)} failed")
    for tag, outcome in sorted(failed.items()):
        log(f"  FAILED {tag}: {outcome}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
