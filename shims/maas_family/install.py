#!/usr/bin/env python3
"""Install the shared-layer shim into MaAS and DAAO.

Idempotent: every edit is guarded by a marker, so re-running after a repo update
does not duplicate anything. Only three files per repo are touched (the shim
itself, the evaluator's dispatch table, and the experiment config table) and the
author code paths are left alone, so the methods still run their own logic --
only the data source and the grading are redirected to the shared layer.

    python shims/maas_family/install.py [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exec_guard_patch  # noqa: E402
import dedent_patch  # noqa: E402
import namespace_patch  # noqa: E402
from maas_family import patch_providers  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHIM_SOURCE = Path(__file__).resolve().parent / "shared_shim.py"
DATA = ROOT / "shared" / "data"

MARKER = "# --- shared-layer shim (agent_wf_v2) ---"

REPOS = {
    "maas": ROOT / "third_party" / "maas" / "maas",
    "daao": ROOT / "third_party" / "daao" / "daao",
}

# dataset key -> (shared eval file, shared search file, question_type)
#
# Keys are prefixed SHARED_ because the evaluator derives its data path as
# f"{dataset.lower()}_{test|train}.jsonl", and DAAO ships its own
# drop_test.jsonl / mbpp_test.jsonl / mbpp_train.jsonl. Unprefixed keys silently
# replaced those author files with links to our splits.
DATASETS = {
    "SHARED_MATH": ("math.jsonl", "math_search.jsonl", "math"),
    # AMC adopted verbatim from FlowBank's shipped split (2026-08-24).
    "SHARED_AMC": ("amc.jsonl", "amc_search.jsonl", "math"),
    "SHARED_MBPP": ("mbpp.jsonl", "mbpp_search.jsonl", "code"),
    "SHARED_DROP": ("drop.jsonl", "drop_search.jsonl", "math"),
    "SHARED_MMLUPRO": ("mmlu_pro.jsonl", "mmlu_pro_search.jsonl", "math"),
}

# Files an earlier revision of this installer created with colliding names.
LEGACY_LINKS = ("math500_test.jsonl", "math500_train.jsonl", "amc_test.jsonl",
                "amc_train.jsonl", "mmlupro_test.jsonl", "mmlupro_train.jsonl",
                "drop_test.jsonl", "drop_train.jsonl", "mbpp_test.jsonl", "mbpp_train.jsonl")

MATH_OPERATORS = ["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble",
                  "Programmer", "SelfRefine", "EarlyStop"]
CODE_OPERATORS = ["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble",
                  "Test", "SelfRefine", "EarlyStop"]

problems: list[str] = []


def report(ok: bool, message: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {message}")
    if not ok:
        problems.append(message)


def install_shim(pkg: Path) -> None:
    target = pkg / "ext" / "maas" / "benchmark" / "shared_shim.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SHIM_SOURCE, target)
    report(target.exists(), f"shim -> {target.relative_to(ROOT)}")


def patch_evaluator(pkg: Path) -> None:
    path = pkg / "ext" / "maas" / "scripts" / "evaluator.py"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        report(True, f"evaluator already patched ({path.relative_to(ROOT)})")
        return

    package = pkg.name
    anchor = '            "HumanEval": HumanEvalBenchmark,\n        }'
    if anchor not in text:
        report(False, f"anchor not found in {path.relative_to(ROOT)}")
        return

    injection = (
        '            "HumanEval": HumanEvalBenchmark,\n'
        f'        }}\n'
        f'        {MARKER}\n'
        f'        from {package}.ext.maas.benchmark.shared_shim import SHARED_DATASET_CONFIGS\n'
        '        self.dataset_configs.update(SHARED_DATASET_CONFIGS)'
    )
    text = text.replace(anchor, injection, 1)

    # DatasetType is a typing Literal and not enforced at runtime, but keeping it
    # accurate stops type checkers and readers from treating the new keys as bugs.
    old_literal = 'DatasetType = Literal["HumanEval", "GSM8K", "MATH"]'
    if old_literal in text:
        keys = ", ".join(f'"{k}"' for k in DATASETS)
        text = text.replace(old_literal,
                            f'DatasetType = Literal["HumanEval", "GSM8K", "MATH", {keys}]', 1)

    path.write_text(text, encoding="utf-8")
    report(MARKER in path.read_text(encoding="utf-8"), f"evaluator patched ({path.relative_to(ROOT)})")



def _operators_for(pkg: Path, key: str, qtype: str) -> list[str]:
    """Take the operator list from the repo's own template/operator.json.

    Hardcoding it copied a bug straight out of DAAO: its experiment_configs.py
    lists seven operators including EarlyStop, but DAAO deleted EarlyStop from
    operator.json and commented out its class, so the name list the graph builds
    is shorter than the embedding matrix the optimizer builds. Whenever the
    controller samples the last index the run dies with IndexError -- three of
    eight samples in a smoke run -- and DAAO's broad `except` reports it as a
    score of 0. MaAS's json still has seven, so the lists genuinely differ per
    repo and must be read, not assumed.
    """
    source = WORKSPACE_SOURCE[key]
    path = pkg / "ext" / "maas" / "scripts" / "optimized" / source / "train" / "template" / "operator.json"
    try:
        names = list(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        names = []
    if names:
        return names
    report(False, f"{key}: could not read {path.name}, falling back to the default list")
    return CODE_OPERATORS if qtype == "code" else MATH_OPERATORS


def patch_experiment_configs(pkg: Path) -> None:
    path = pkg / "ext" / "maas" / "benchmark" / "experiment_configs.py"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        report(True, f"experiment_configs already patched ({path.relative_to(ROOT)})")
        return

    lines = [f"\n{MARKER}"]
    for key, (_eval_file, _search_file, qtype) in DATASETS.items():
        operators = _operators_for(pkg, key, qtype)
        lines.append(
            f'EXPERIMENT_CONFIGS["{key}"] = ExperimentConfig(\n'
            f'    dataset="{key}",\n'
            f'    question_type="{qtype}",\n'
            f'    operators={operators!r},\n'
            f')'
        )
    path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    report(MARKER in path.read_text(encoding="utf-8"), f"experiment_configs patched ({path.relative_to(ROOT)})")


CONFIG_TEMPLATE = """# Generated by shims/maas_family/install.py -- do not hand-edit.
#
# Optimizer and executor are the same local Qwen3-8B behind the experiment proxy,
# which forces temperature=0, disables thinking, recovers truncated replies and
# records per-namespace token accounting. The URL path is the namespace, so
# search traffic and held-out test traffic stay separable.
llm:
  api_type: "openai"
  model: "Qwen/Qwen3-8B"
  base_url: "http://127.0.0.1:18080/{namespace}/v1"
  api_key: "local"
  temperature: 0
  # Must stay false: the experiment proxy needs the whole response body to strip
  # leaked thinking, recover truncation and account tokens. MaAS defaults this to
  # true, and a streamed request against a non-streaming body reads as an empty
  # string -- which surfaced as pydantic "Missing fields" errors and zero-gradient
  # training batches rather than as any kind of connection error.
  stream: false
models:
  qwen3-8b:
    api_type: "openai"
    model: "Qwen/Qwen3-8B"
    base_url: "http://127.0.0.1:18080/{namespace}/v1"
    api_key: "local"
    temperature: 0
    stream: false
"""


def write_config(repo_root: Path, label: str) -> None:
    """Point the repo at the proxy.

    Kept in the installer rather than written by hand: an ad-hoc config is not
    reproducible, and a `git checkout` that reverts other edits silently takes
    the credentials with it, leaving the repo to fail with an empty api_key.
    """
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config2.yaml"
    path.write_text(CONFIG_TEMPLATE.format(namespace=f"train/{label}"), encoding="utf-8")
    report("base_url" in path.read_text(encoding="utf-8"), f"config -> {path.relative_to(ROOT)}")


# Each dataset needs a starting workspace under scripts/optimized/<KEY>/<split>/:
# graph.py plus a template/ directory of operator implementations and prompts.
# The repo ships three -- MATH, GSM8K, HumanEval -- so the shared datasets seed
# from the closest one. There is no QA template, so DROP and MMLU-Pro inherit
# MATH's; that is a declared substitution, matching the operator lists already
# registered in EXPERIMENT_CONFIGS (Generate / GenerateCoT / MultiGenerateCoT /
# ScEnsemble / Programmer / SelfRefine / EarlyStop).
WORKSPACE_SOURCE = {
    "SHARED_MATH": "MATH",
    "SHARED_AMC": "MATH",
    "SHARED_DROP": "MATH",
    "SHARED_MMLUPRO": "MATH",
    "SHARED_MBPP": "HumanEval",
}


# --- WITHDRAWN: the ScEnsemble degradation patch ---
#
# An earlier revision rewrote ScEnsemble to fall back to solutions[-1] whenever
# its XML field extraction failed, on the theory that an 8B executor could not
# reliably name a candidate letter. That was one of four wrong diagnoses of the
# same symptom. The real cause was this proxy rewriting stream=True to False:
# MaAS's SSE parser found no `data:` lines in a plain JSON body and returned "",
# which surfaced as pydantic ValidationError inside ScEnsemble.
#
# Re-running shims/maas_family/probe_scensemble.py after the SSE passthrough fix
# returns {'solution_letter': 'A'} with no error, and the ValidationError count
# goes from 26 (maas_smoke6, before the fix) to 0 (maas_final and daao_smoke,
# after it). The patch was therefore unnecessary -- and it was a modification to
# the operator's behaviour on a path the paper does not define, which is exactly
# the kind of change this comparison should not be making without evidence.
#
# It is withdrawn rather than kept-but-disabled: the author operator is restored
# verbatim. A genuine failure is still visible, because shared_shim logs the
# sample failure with a traceback naming ScEnsemble, so the rate stays measurable
# without altering what the operator does.


SAFE_FORMAT_HELPER = '''

# --- shared-layer shim: brace-safe prompt substitution ---
# The shipped templates embed LaTeX demonstration examples such as \\boxed{-2},
# \\frac{1}{3} and \\boxed{144\\pi}. str.format() reads every one of those as a
# replacement field, so GENERATE_COT_PROMPT.format(...) raises KeyError("-2") on
# *any* input -- 40 of 40 MATH problems reproduce it. Two operators (GenerateCoT,
# MultiGenerateCoT) therefore never worked, and MaAS's own broad `except` in
# evaluate_problem masked it as a score of 0.
#
# This substitutes only the named placeholders and leaves every other brace as
# literal text, which is what the templates intend.
def _shim_safe_format(_template, **kwargs):
    out = _template
    for _key, _value in kwargs.items():
        out = out.replace("{" + _key + "}", str(_value))
    return out
'''


def patch_prompt_formatting(pkg: Path) -> None:
    """Route operator prompt building through a brace-safe substitution."""
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    if not base.exists():
        report(False, "cannot patch prompt formatting: optimized/ missing")
        return
    pattern = re.compile(r"([A-Z_]+_PROMPT)\.format\(", )
    patched = already = 0
    for path in sorted(base.glob("*/*/template/operator.py")):
        text = path.read_text(encoding="utf-8")
        if "_shim_safe_format" in text:
            already += 1
            continue
        replaced, count = pattern.subn(r"_shim_safe_format(\1, ", text)
        if count == 0:
            continue
        path.write_text(replaced + SAFE_FORMAT_HELPER, encoding="utf-8")
        patched += 1
    report(patched + already > 0,
           f"prompt formatting made brace-safe ({patched} patched, {already} already)")


# Backport of FlowBank's own MMLU-Pro ScEnsemble (its commit dde948d, file
# DiverseFlow/workspace/MMLU_Pro/workflows/template/operator.py).
#
# The operator labels the candidate solutions A, B, C and asks the model which
# letter is best. That is unambiguous for MATH, GSM8K or HumanEval, whose answers
# are never a bare letter -- but MMLU-Pro is ten-way multiple choice with options
# (A) to (J), so the two letter spaces collide. The model replies "E" meaning the
# question's option E; the operator reads it as the fifth candidate, of which
# there are three, and `solutions[answer_mapping["E"]]` raises KeyError. Our shim
# then records the sample as a zero-gradient placeholder, so it is not merely
# graded wrong, it is discarded: measured 29 of ~56 samples for MaAS and 43 of ~80
# for DAAO on MMLU-Pro, i.e. over half the training signal.
#
# FlowBank ships in this comparison with its author's fix already applied, so
# leaving MaAS/DAAO/AFlow unpatched is not neutrality -- it hands one method a
# robustness advantage on a dataset *we* added to repos that never supported it.
# The numeric labelling and the fallback ladder below are lifted from that file
# rather than invented here.
#
# Two deliberate limits:
#  * the numeric relabelling applies only on MMLU-Pro, since that is the only
#    dataset where the letter spaces overlap;
#  * the out-of-range guard applies everywhere, which cannot change a working run
#    -- it only intercepts the input that currently raises.
# Version marker. Without it "already patched" keeps a stale block in place: the
# helper _shim_code_dataset was added in v2, and every file carrying v1 was skipped
# as done, so the graph imported an operator module that had no such helper. The
# same omission cost two earlier fixes -- shims/aflow/install.py gained this
# mechanism and this file did not.
SCENSEMBLE_PATCH_MARKER = "# --- shared-layer shim (agent_wf_v2) --- scensemble v2"
SC_ENSEMBLE_NUMERIC = '''

# --- shared-layer shim (agent_wf_v2) --- scensemble v2
# ScEnsemble label-space fix, backported verbatim in behaviour from FlowBank's
# own MMLU-Pro operator (author commit dde948d). See shims/maas_family/install.py
# for why this is applied to MaAS/DAAO as well.
import os as _shim_os
import re as _shim_re
import sys as _shim_sys
from pydantic import BaseModel as _ShimBaseModel, Field as _ShimField

SC_ENSEMBLE_NUMERIC_PROMPT = """Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_number" field, output ONLY the numeric ID (1, 2, 3, etc.) corresponding to the most consistent solution. Do NOT output a letter - output the number only.
"""


class ScEnsembleNumericOp(_ShimBaseModel):
    thought: str = _ShimField(default="", description="The thought of the most consistent solution.")
    solution_number: str = _ShimField(default="", description="The number of the most consistent solution (1, 2, 3, etc.).")


def _shim_code_dataset() -> str:
    """Which code benchmark's public tests the Test operator should look up.

    The seeded SHARED_MBPP workspace imports this template from the authors'
    HumanEval workspace -- that is how MaAS lays out workspaces, and it is why
    patching SHARED_MBPP's own copy changed nothing: that copy is never imported.
    The hardcoded `dataset="HumanEval"` therefore sent MBPP function names to
    humaneval_public_test.jsonl, which MaAS does not ship at all (1,020 samples
    discarded) and which DAAO ships without those names (255 discarded).

    Decided at runtime for the same reason as the label space below: one template
    serves several datasets, so the answer cannot be baked in at install time.
    """
    marker = (_shim_os.getenv("SHIM_DATASET", "") or " ".join(_shim_sys.argv)).upper()
    return "SHARED_MBPP" if "MBPP" in marker else "HumanEval"


def _shim_on_mmlu_pro() -> bool:
    """Whether this process is running the dataset whose answers are letters.

    Read from the environment first (sweep.py sets SHIM_DATASET per job) and from
    argv second, so a manual `optimize.py --dataset SHARED_MMLUPRO` outside the
    sweep still gets the fix instead of silently reverting to letters.
    """
    marker = _shim_os.getenv("SHIM_DATASET", "") or " ".join(_shim_sys.argv)
    marker = marker.upper()
    return "MMLUPRO" in marker or "MMLU_PRO" in marker


class ScEnsemble(ScEnsemble):  # noqa: F811 - wraps the author class defined above
    async def __call__(self, solutions, problem: str):
        numeric = _shim_on_mmlu_pro()
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            label = str(index + 1) if numeric else chr(65 + index)
            answer_mapping[label] = index
            solution_text += f"{label}: \\n{str(solution)}\\n\\n\\n"

        if numeric:
            prompt = _shim_safe_format(SC_ENSEMBLE_NUMERIC_PROMPT,
                                       question=problem, solutions=solution_text)
            response = await self._fill_node(ScEnsembleNumericOp, prompt, mode="xml_fill")
            answer = str(response.get("solution_number", "")).strip()
        else:
            prompt = _shim_safe_format(SC_ENSEMBLE_PROMPT,
                                       problem=problem, solutions=solution_text)
            response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")
            answer = str(response.get("solution_letter", "")).strip().upper()

        if answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}

        # Fallback ladder, as in the FlowBank original: pull a label out of a
        # wordier reply, then fall back to the first solution. Reaching the last
        # step is logged rather than silent, because it is a real degradation of
        # the ensemble -- the selection stops being a choice.
        for token in _shim_re.findall(r"\\d+" if numeric else r"[A-Z]", answer):
            if token in answer_mapping:
                return {"response": solutions[answer_mapping[token]]}
        _shim_sys.stderr.write(
            f"[shared_shim] ScEnsemble got {answer!r}, not one of "
            f"{sorted(answer_mapping)}; falling back to the first solution\\n")
        return {"response": solutions[0]}
'''


def patch_mbpp_test_dataset(pkg: Path) -> None:
    """Make the Test operator's benchmark choice a runtime decision.

    Every seeded workspace imports its operators from the *source* workspace, not
    from its own copy: SHARED_MATH/AMC/DROP/MMLUPRO all import
    `optimized.MATH.train.template.operator`, and SHARED_MBPP imports
    `optimized.HumanEval.train.template.operator`. So the file that must change is
    the authors' HumanEval template, and changing SHARED_MBPP's copy -- which an
    earlier revision of this function did -- has no effect whatsoever, because
    nothing imports it. The stack trace is what settled it:

        SHARED_MBPP/train/graph.py:79
          -> optimized/HumanEval/train/template/operator.py:144
             FileNotFoundError: 'maas/ext/maas/data/humaneval_public_test.jsonl'

    Rewriting the call to ask `_shim_code_dataset()` keeps HumanEval correct for
    HumanEval (the helper returns "HumanEval" unless the dataset says MBPP) while
    making it correct for MBPP too, and does it in the one file both workspaces
    share instead of in five copies.
    """
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    if not base.exists():
        report(False, "cannot patch the Test operator: optimized/ missing")
        return
    patched = already = 0
    for path in sorted(base.glob("*/*/template/operator.py")):
        text = path.read_text(encoding="utf-8")
        upgraded = text.replace(
            'return "MBPP" if "MBPP" in marker else "HumanEval"',
            'return "SHARED_MBPP" if "MBPP" in marker else "HumanEval"',
        )
        if upgraded != text:
            path.write_text(upgraded, encoding="utf-8")
            text = upgraded
        # The condition is about the *call site*, not about the helper existing.
        # Testing for "_shim_code_dataset()" skipped all sixteen files the moment
        # the ScEnsemble block (which defines that helper) was re-appended, leaving
        # twelve call sites still hardcoded while reporting "already done".
        if not re.search(r'dataset=(?:"|\')(?:HumanEval|MBPP)(?:"|\')', text):
            already += 1
            continue
        replaced = re.sub(r'dataset=(?:"|\')(?:HumanEval|MBPP)(?:"|\')',
                          "dataset=_shim_code_dataset()", text)
        if replaced == text:
            continue
        path.write_text(replaced, encoding="utf-8")
        verify_syntax(path)
        patched += 1
    report(patched + already > 0,
           f"Test operator picks its benchmark at runtime ({patched} patched, "
           f"{already} already)")


def write_mbpp_public_tests(pkg: Path) -> None:
    """Write one task-aware public-test record per shared MBPP item."""
    import re as _re

    data_dir = pkg / "ext" / "maas" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "shared_mbpp_public_test.jsonl"
    written = 0
    entry_counts: dict[str, int] = {}
    with target.open("w", encoding="utf-8") as handle:
        for name in ("mbpp.jsonl", "mbpp_search.jsonl"):
            source = DATA / name
            if not source.exists():
                report(False, f"missing shared split {name}")
                continue
            with source.open(encoding="utf-8") as source_handle:
                for line in source_handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    entry, tests = row.get("entry_point"), row.get("test_list") or []
                    if not entry or not tests:
                        continue
                    entry_counts[entry] = entry_counts.get(entry, 0) + 1
                    renamed = [
                        _re.sub(rf"\b{_re.escape(entry)}\b", "candidate", test)
                        for test in tests
                    ]
                    task = (f"{row.get('prompt', '')}\n\n"
                            "Your code must pass these tests:\n" + "\n".join(tests))
                    handle.write(json.dumps({
                        "uid": row.get("uid", ""),
                        "entry_point": entry,
                        "prompt": row.get("prompt", ""),
                        "task": task,
                        "test": renamed,
                    }, ensure_ascii=False) + "\n")
                    written += 1
    report(written > 0,
           f"MBPP public tests -> {target.relative_to(ROOT)} "
           f"({written} items, "
           f"{sum(n - 1 for n in entry_counts.values() if n > 1)} shared names)")


def patch_mbpp_test_identity(pkg: Path) -> None:
    """Resolve shared MBPP tests by entry point and the current task text."""
    utils_path = pkg / "ext" / "maas" / "scripts" / "utils.py"
    text = utils_path.read_text(encoding="utf-8")
    old_sig = (
        "def extract_test_cases_from_jsonl(entry_point: str, "
        "dataset: CodeDataset = CodeDataset.HUMAN_EVAL):"
    )
    new_sig = old_sig[:-2] + ", problem: str = None):"
    if old_sig in text:
        text = text.replace(old_sig, new_sig, 1)

    shared_branch = '''    if dataset == "SHARED_MBPP":
        file_path = "{package}/ext/maas/data/shared_mbpp_public_test.jsonl"
        hardcoded_cases = {{}}
    elif dataset == CodeDataset.HUMAN_EVAL.value:'''.format(package=pkg.name)
    author_branch = "    if dataset == CodeDataset.HUMAN_EVAL.value:"
    if author_branch in text and 'dataset == "SHARED_MBPP"' not in text:
        text = text.replace(author_branch, shared_branch, 1)

    text = text.replace(
        "    if entry_point in hardcoded_cases:\n"
        "        return hardcoded_cases[entry_point]",
        "    if problem is None and entry_point in hardcoded_cases:\n"
        "        return hardcoded_cases[entry_point]",
        1,
    )
    old_lookup = '''    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("entry_point") == entry_point:
                return data.get("test")

    return None'''
    new_lookup = '''    matches = []
    task_matches = []
    prompt_matches = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("entry_point") != entry_point:
                continue
            task = str(data.get("task") or "")
            prompt = str(data.get("prompt") or "")
            if problem is not None and task and str(problem).startswith(task):
                task_matches.append(data.get("test"))
            if problem is not None and prompt and str(problem).startswith(prompt):
                prompt_matches.append(data.get("test"))
            matches.append(data.get("test"))

    if len(task_matches) == 1:
        return task_matches[0]
    if len(prompt_matches) == 1:
        return prompt_matches[0]
    return matches[0] if len(matches) == 1 else None'''
    if old_lookup in text:
        text = text.replace(old_lookup, new_lookup, 1)
    old_task_lookup = '''    matches = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("entry_point") != entry_point:
                continue
            prompt = str(data.get("prompt") or "")
            if problem is not None and prompt and str(problem).startswith(prompt):
                return data.get("test")
            matches.append(data.get("test"))

    return matches[0] if len(matches) == 1 else None'''
    if old_task_lookup in text:
        text = text.replace(old_task_lookup, new_lookup, 1)
    utils_path.write_text(text, encoding="utf-8")
    verify_syntax(utils_path)

    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    touched = 0
    for path in sorted(base.glob("*/*/template/operator.py")):
        body = path.read_text(encoding="utf-8")
        updated = body.replace(
            "def exec_code(self, solution, entry_point):",
            "def exec_code(self, problem, solution, entry_point):",
        )
        updated = updated.replace(
            "extract_test_cases_from_jsonl(entry_point, dataset=_shim_code_dataset())",
            "extract_test_cases_from_jsonl(entry_point, dataset=_shim_code_dataset(), problem=problem)",
        )
        updated = updated.replace(
            "self.exec_code(solution, entry_point)",
            "self.exec_code(problem, solution, entry_point)",
        )
        if updated != body:
            path.write_text(updated, encoding="utf-8")
            verify_syntax(path)
            touched += 1

    final = utils_path.read_text(encoding="utf-8")
    report('dataset == "SHARED_MBPP"' in final and "len(task_matches) == 1" in final,
           "MBPP public-test lookup accepts task identity")
    code_templates = [p for p in base.glob("*/*/template/operator.py")
                      if "class Test" in p.read_text(encoding="utf-8")]
    report(bool(code_templates) and all("problem=problem" in
                                       p.read_text(encoding="utf-8")
                                       for p in code_templates),
           f"MBPP Test operators pass task identity ({touched} file(s) updated)")


def patch_scensemble_labels(pkg: Path) -> None:
    """Give MaAS/DAAO the label-space guard FlowBank already ships with."""
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    if not base.exists():
        report(False, "cannot patch ScEnsemble: optimized/ missing")
        return
    patched = already = upgraded = 0
    for path in sorted(base.glob("*/*/template/operator.py")):
        text = path.read_text(encoding="utf-8")
        if SCENSEMBLE_PATCH_MARKER in text:
            already += 1
            continue
        if "ScEnsembleNumericOp" in text:
            # An older version of this block. Truncate at its marker and re-append
            # rather than skipping -- skipping is what left 16 files without the
            # _shim_code_dataset helper the Test operator now calls.
            text = text.split("# --- shared-layer shim (agent_wf_v2) ---")[0].rstrip() + "\n"
            upgraded += 1
        elif "class ScEnsemble" not in text:
            continue
        # Appended, not edited in place: the author's class stays exactly as
        # written and the subclass below shadows it, so a diff against upstream
        # shows an addition rather than a rewrite of their logic.
        path.write_text(text + SC_ENSEMBLE_NUMERIC, encoding="utf-8")
        verify_syntax(path)
        patched += 1
    report(patched + already > 0,
           f"ScEnsemble label-space guard installed ({patched} patched"
           + (f", {upgraded} upgraded from an older block" if upgraded else "")
           + f", {already} already current)")


# Two distinct repairs, appended to the prompt modules rather than edited into
# them, so the author's text stays visible above and every change is legible in a
# diff against archive/prompt_baseline.
#
# (1) DEFECT REPAIR, applied on every dataset. The shipped templates are
#     triple-quoted non-raw strings containing LaTeX, so `\boxed` compiles to the
#     single byte 0x08 followed by "oxed". The file reads correctly in an editor
#     and in `git show`; only the runtime value is wrong. Measured with
#     scan_escape_corruption.py: 92 occurrences in MaAS and 92 in DAAO, none in
#     the other five methods. So MaAS and DAAO alone were instructing the model to
#     "present the final answer enclosed in <BS>oxed{} LaTeX notation" while the
#     grader looked for \boxed -- a formatting handicap unrelated to the workflow
#     under study. This is not a design change: it restores the literal text the
#     author wrote.
#
# (2) DATASET ADAPTATION, applied only on DROP, MMLU-Pro and MBPP. All five
#     SHARED_* workspaces import their prompts from the authors' MATH (or
#     HumanEval) template -- the copies inside SHARED_DROP/ and friends are never
#     imported, which is why editing them changes nothing. So the two datasets we
#     added were being told to "solve the given mathematical problem" and to
#     "present the final answer enclosed in \boxed{}", while the task text asks for
#     a passage short answer or an option letter. Only task identity is rebound here:
#     which task this is, what the answer should look like, and the worked example.
#     Roles keep their shape, the numbered guidelines keep their count and order,
#     the output-format machinery, debate structure, SC_ENSEMBLE_PROMPT and
#     SELFREFINE_PROMPT are untouched, and MATH/AMC see the author's text verbatim
#     apart from repair (1).
PROMPT_ADAPT_MARKER = "# --- shared-layer shim (agent_wf_v2) --- prompt task adaptation v4"
PROMPT_ADAPT_BLOCK = r'''

# --- shared-layer shim (agent_wf_v2) --- prompt task adaptation v4
# See shims/maas_family/install.py for why this is appended rather than edited in.
# Every override below is a raw string: the mis-escaped LaTeX repaired just above
# is exactly the bug that non-raw prompt literals cause.
import os as _shim_ap_os
import sys as _shim_ap_sys


def _shim_ap_task() -> str:
    """Which dataset this process is running.

    Read from the environment first (sweep.py exports SHIM_DATASET per job) and
    from argv second, so a manual `optimize.py --dataset SHARED_DROP` outside the
    sweep is adapted too instead of silently falling back to the maths wording.
    MMLU-Pro is tested before MATH because "SHARED_MMLUPRO" contains neither, and
    DROP before MBPP only for readability -- the markers are disjoint.
    """
    marker = (_shim_ap_os.getenv("SHIM_DATASET", "") or " ".join(_shim_ap_sys.argv)).upper()
    for _needle, _name in (("MMLUPRO", "mmlu_pro"), ("MMLU_PRO", "mmlu_pro"),
                           ("DROP", "drop"), ("MBPP", "mbpp"), ("AMC", "amc")):
        if _needle in marker:
            return _name
    return "math"


_SHIM_AP_TASK = _shim_ap_task()

# (1) Repair, unconditional. Only these three tokens are rewritten, not every
# control character: 0x09 is also an ordinary tab and 0x0a an ordinary newline in
# prompt text, so a blanket rule would corrupt legitimate whitespace. The
# installer's --check asserts that nothing else remains.
for _shim_ap_name in [_n for _n in list(globals()) if _n.endswith("_PROMPT")]:
    _shim_ap_text = globals()[_shim_ap_name]
    if isinstance(_shim_ap_text, str):
        globals()[_shim_ap_name] = (_shim_ap_text
                                    .replace("\x08oxed", "\\boxed")
                                    .replace("\x0crac", "\\frac")
                                    .replace("\x09imes", "\\times"))
# Deleted, not just left to fall out of use: _shim_ap_text still holds the
# *pre-repair* text of the last constant, and it would sit in the module namespace
# where anything walking globals() -- including this project's own scanners -- reads
# it as a live prompt. Measured: 46 phantom hits per repo before this line existed.
del _shim_ap_name, _shim_ap_text

# (2) Adaptation, per dataset.
if _SHIM_AP_TASK == "drop":
    # Gated on the task alone. Gating it on "GENERATE_SOLUTION_PROMPT" in globals()
    # -- as an earlier revision did -- skipped this whole block inside op_prompt.py,
    # which is where GENERATE_COT_PROMPT and PYTHON_CODE_VERIFIER_PROMPT live: the
    # live transcripts then showed 53% of DAAO's DROP prompts still carrying the
    # GSM8K worked examples and "based on a given mathematical problem". The names
    # this defines in the module that does not use them are unread and harmless;
    # skipping the ones that ARE used was not.
    GENERATE_SOLUTION_PROMPT = r"""
Please answer the given reading comprehension question about the passage step by step. Follow these guidelines:

1. State the question clearly.
2. Outline the approach and identify the parts of the passage that bear on it.
3. Provide the detailed derivation, quoting the figures, dates or names the passage gives.
4. Explain each step of your reasoning.
5. Present the final answer on a last line of the form 'Answer: <answer>', where <answer> is the concise answer (a span, number, date, or list as appropriate).
6. Quote the passage exactly when the question asks for a textual span.

Your solution should be thorough, faithful to the passage, and easy to understand.
"""
    MATH_SOLUTION_PROMPT = GENERATE_SOLUTION_PROMPT
    REFINE_ANSWER_PROMPT = r"""
Given the reading comprehension question, its passage and the output from the code execution, please provide a well-formatted and detailed answer. Follow these guidelines:

1. Begin with a clear statement of the question.
2. Explain the approach and which parts of the passage were used.
3. Show the step-by-step derivation, quoting the figures the passage gives.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer on a last line of the form 'Answer: <answer>', where <answer> is the concise answer (a span, number, date, or list as appropriate).
6. Quote the passage exactly when the question asks for a textual span.

Your response should be comprehensive, faithful to the passage, and easy to follow.
"""
    SOLUTION_PROMPT = r"""
Provide a comprehensive, step-by-step answer to the given reading comprehension question. Your response should include:

1. A clear restatement of the question.
2. An explanation of what the passage says on the point at issue.
3. A detailed, logical progression of steps leading to the answer.
4. Clear explanations for each step, including the reasoning behind it.
5. All figures and dates quoted exactly as the passage gives them.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked on a last line of the form 'Answer: <answer>', where <answer> is the concise answer (a span, number, date, or list as appropriate).
8. A brief explanation of the significance of the result, if relevant.

Ensure your answer is rigorous, easy to follow, and faithful to the passage.
"""
    DETAILED_SOLUTION_PROMPT = SOLUTION_PROMPT
    MATH_SOLVE_PROMPT = r"""
You are a highly skilled reading comprehension analyst tasked with answering a question about a passage. Follow these steps carefully:

1. Read and understand the passage and the question thoroughly.
2. Identify all key figures, dates, names and relationships the passage states.
3. Determine what the question asks for: a span to quote, a count, or an arithmetic result over stated figures.
4. Work the answer out step-by-step, showing all your work clearly.
5. Double-check your reading and any arithmetic at each step.
6. Provide a clear and concise final answer.
7. Verify your answer against the passage and any arithmetic against the stated figures.

Format your answer as follows:
- Quote textual spans exactly as the passage writes them.
- Show each step of your reasoning clearly.
- Clearly state your final answer at the end of your solution.
- Express numerical answers as precise values (avoid rounding unless specified).
- Ensure that your final answer is concise: a span, number, date, or list as appropriate.
- Do not include any explanatory text on the final answer line.

For example, if the final answer is 57, your response should end with just:
Answer: 57

Here's the question to answer:

"""
    if "PYTHON_CODE_VERIFIER_PROMPT" in globals():
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "based on a given mathematical problem and output the answer",
            "based on a given reading comprehension question and its passage, and output the answer")
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "Implement the calculation steps described in the problem.",
            "Implement the counting or arithmetic steps over the figures the passage states.")
    if "GENERATE_COT_PROMPT" in globals():
        GENERATE_COT_PROMPT = r"""
Reading Comprehension Reasoning Instruction
{instruction}

Current Problem: {input}

Demonstration Examples (DROP style):

1. Passage: "The city council approved 14 permits in March and 9 permits in April."
   Question: How many permits were approved in total over the two months?
   Analysis:
   Locate both figures in the passage: 14 in March, 9 in April
   The question asks for the total, so add them: 14 + 9 = 23
   Answer: 23

2. Passage: "Ferrer won the 2007 final, and Nadal won in 2008 and 2009."
   Question: Who won the final the year before Nadal's first title?
   Analysis:
   Nadal's first title is 2008, so the year before is 2007
   The passage names the 2007 winner: Ferrer
   Copy the span exactly as written
   Answer: Ferrer

Solution Protocol:
1. Parse the passage and the question carefully
2. Identify the spans and figures that bear on the question
3. Perform the stepwise derivation over those figures
4. Verify intermediate results against the passage
5. Present the final answer on a last line of the form 'Answer: <answer>'

Step-by-Step Analysis:
"""

if _SHIM_AP_TASK == "mmlu_pro":
    # Same reasoning as the DROP block above.
    GENERATE_SOLUTION_PROMPT = r"""
Please solve the given multiple-choice question step by step. Follow these guidelines:

1. State the question clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide the detailed reasoning, using LaTeX notation for any mathematical expressions.
4. Explain each step of your reasoning, including why the other options are wrong.
5. Present the final answer on a last line of the form 'Answer: (X)', where X is a single option letter.
6. Choose exactly one of the listed options.

Your solution should be thorough, well reasoned, and easy to understand.
"""
    MATH_SOLUTION_PROMPT = GENERATE_SOLUTION_PROMPT
    REFINE_ANSWER_PROMPT = r"""
Given the multiple-choice question and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the question.
2. Explain the approach and any formulas or concepts used.
3. Show the step-by-step reasoning, using LaTeX notation for any mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer on a last line of the form 'Answer: (X)', where X is a single option letter.
6. Choose exactly one of the listed options.

Your response should be comprehensive, rigorous, and easy to follow.
"""
    SOLUTION_PROMPT = r"""
Provide a comprehensive, step-by-step solution to the given multiple-choice question. Your response should include:

1. A clear restatement of the question.
2. An explanation of the concepts and principles involved.
3. A detailed, logical progression of steps leading to the answer.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked on a last line of the form 'Answer: (X)', where X is a single option letter.
8. A brief explanation of why the remaining options are wrong, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
"""
    DETAILED_SOLUTION_PROMPT = SOLUTION_PROMPT
    MATH_SOLVE_PROMPT = r"""
You are a highly skilled expert tasked with answering a multiple-choice question. Follow these steps carefully:

1. Read and understand the question and every option thoroughly.
2. Identify all key information, variables, and relationships.
3. Determine the appropriate concepts, formulas, or equations to use.
4. Work the question out step-by-step, showing all your work clearly.
5. Double-check your reasoning and calculations at each step.
6. Provide a clear and concise final answer.
7. Verify your answer by checking the remaining options can be ruled out.

Format your answer as follows:
- Use LaTeX notation for mathematical expressions where appropriate.
- Show each step of your solution process clearly.
- Clearly state your final answer at the end of your solution.
- Give exactly one option letter, chosen from the options listed with the question.
- Ensure that your final answer is a single option letter without any units or additional text.
- Do not include any explanatory text with your final answer, just the letter itself.

For example, if the correct option is C, your response should end with just:
Answer: (C)

Here's the question to answer:

"""
    if "PYTHON_CODE_VERIFIER_PROMPT" in globals():
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "based on a given mathematical problem and output the answer",
            "based on a given multiple-choice question and output the answer")
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "Implement the calculation steps described in the problem.",
            "Implement the calculation steps needed to decide between the options.")
    if "GENERATE_COT_PROMPT" in globals():
        GENERATE_COT_PROMPT = r"""
Multiple-Choice Reasoning Instruction
{instruction}

Current Problem: {input}

Demonstration Examples (MMLU-Pro style):

1. Question: Which unit measures electric current?
   Options: (A) volt (B) ampere (C) ohm (D) watt
   Analysis:
   Current is charge per unit time, whose SI unit is the ampere
   Volt measures potential, ohm resistance, watt power, so all three are ruled out
   Answer: (B)

2. Question: A body accelerates from rest at 3 m/s^2 for 4 s. What is its final speed?
   Options: (A) 7 m/s (B) 10 m/s (C) 12 m/s (D) 24 m/s
   Analysis:
   From rest, $v = at$
   Substitute values: $v = 3 \times 4 = 12$ m/s
   Match against the options: 12 m/s is option C
   Answer: (C)

Solution Protocol:
1. Parse the question and every option carefully
2. Identify the relevant concepts
3. Perform the stepwise derivation
4. Rule out the remaining options
5. Present the final answer on a last line of the form 'Answer: (X)'

Step-by-Step Analysis:
"""

if _SHIM_AP_TASK == "mbpp":
    # The MBPP cell imports the authors' HumanEval template, so both constants
    # name the wrong benchmark to the model. Nothing else in this template is
    # HumanEval-specific: the operator-level prompts are generic.
    for _shim_ap_name in ("IMPROVE_CODE_PROMPT", "GENERATE_CODE_PROMPT"):
        if _shim_ap_name in globals() and isinstance(globals()[_shim_ap_name], str):
            globals()[_shim_ap_name] = (globals()[_shim_ap_name]
                                        .replace("HumanEval benchmark", "MBPP benchmark")
                                        .replace("HumanEval dataset", "MBPP dataset"))
'''


def patch_namespace_override(pkg: Path) -> None:
    """Let SHIM_BASE_URL win over config2.yaml, so accounting reaches the cell."""
    path = pkg / "provider" / "llm_provider_registry.py"
    outcome = namespace_patch.patch_file(path, "create_llm_instance", "config")
    report(outcome in ("patched", "already"),
           f"namespace override in create_llm_instance ({outcome})")


def patch_code_execution(pkg: Path) -> None:
    """Bound execution of model-written code with a process that can be killed."""
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    if not base.exists():
        report(False, "cannot patch code execution: optimized/ missing")
        return
    tally: dict[str, int] = {}
    for path in sorted(base.glob("*/*/template/operator.py")):
        outcome = exec_guard_patch.patch_file(path)
        tally[outcome] = tally.get(outcome, 0) + 1
        if outcome == "patched":
            verify_syntax(path)
    report(tally.get("patched", 0) + tally.get("already", 0) > 0,
           "generated-code execution guarded ("
           + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())) + ")")



def patch_indented_code(pkg: Path) -> None:
    """Keep model code that arrives indented, instead of silently dropping it.

    Upstream ``sanitize()`` retains only top-level definitions matching the entry
    point. Qwen3-8B often returns the whole function indented by two spaces --
    measured over 17,141 mbpp code replies from this sweep, 31.3% were reduced to
    an empty string; dedenting first brings that to 1.1% and recovers 5,170 of
    them. The prediction column for those items was empty with no error raised,
    so the loss was invisible in the logs and scored as a wrong answer.
    """
    action_path = pkg / "actions" / "action_node.py"
    sanitize_path = pkg / "utils" / "sanitize.py"
    if not action_path.exists() or not sanitize_path.exists():
        report(False, "cannot patch indented code: action_node.py or sanitize.py missing")
        return
    action_outcome = dedent_patch.patch_file(action_path)
    sanitize_outcome = dedent_patch.patch_sanitize_file(sanitize_path)
    if action_outcome == "patched":
        verify_syntax(action_path)
    if sanitize_outcome == "patched":
        verify_syntax(sanitize_path)
    report(action_outcome in ("patched", "already"),
           f"code_fill preserves indented model code ({action_outcome})")
    report(sanitize_outcome in ("patched", "already"),
           f"sanitize preserves indented model code ({sanitize_outcome})")


def link_sentence_model(pkg: Path) -> None:
    """Expose the shared MiniLM checkout at DAAO's legacy relative path.

    DAAO constructs ``SentenceTransformer('./all-MiniLM-L6-v2')`` while its
    repository directory is the current working directory.  The old server had
    an untracked absolute symlink there, so a fresh clone failed before the
    first request.  Keep the upstream code unchanged and create a portable,
    repository-relative link during installation.
    """
    if pkg.parent.name != "daao":
        return
    link = pkg.parent / "all-MiniLM-L6-v2"
    target = Path("../../shared/models/all-MiniLM-L6-v2")
    if link.is_symlink():
        if os.readlink(link) == str(target):
            report(True, "DAAO MiniLM path is a portable relative link")
        else:
            report(False, f"DAAO MiniLM link points to {os.readlink(link)!r}")
        return
    if link.exists():
        report(False, "DAAO MiniLM path exists but is not the expected symlink")
        return
    link.symlink_to(target)
    report(True, "DAAO MiniLM path linked to shared model")


def patch_prompt_task_wording(pkg: Path) -> None:
    """Repair the mis-escaped LaTeX and adapt task identity per dataset."""
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    if not base.exists():
        report(False, "cannot patch prompt wording: optimized/ missing")
        return
    # An older revision of the block is replaced, not left in place. Treating "some
    # version of this marker is present" as done is the exact failure that kept a
    # stale ScEnsemble block alive through two fixes: the file looked patched and
    # the patch it carried was the previous one.
    stale_prefix = PROMPT_ADAPT_MARKER.rsplit(" v", 1)[0]
    patched = already = upgraded = 0
    for path in sorted(base.glob("*/*/template/prompt.py")) + \
            sorted(base.glob("*/*/template/op_prompt.py")):
        text = path.read_text(encoding="utf-8")
        if PROMPT_ADAPT_MARKER in text:
            already += 1
            continue
        if stale_prefix in text:
            text = text[: text.index(stale_prefix)].rstrip("\n") + "\n"
            upgraded += 1
        else:
            patched += 1
        path.write_text(text + PROMPT_ADAPT_BLOCK, encoding="utf-8")
        verify_syntax(path)
    report(patched + already + upgraded > 0,
           f"prompt task wording adapted ({patched} patched, {upgraded} upgraded, "
           f"{already} already current)")


def patch_test_result_data(pkg: Path) -> None:
    """Fix the author's own test(): it calls create_result_data(round, score)
    while data_utils defines five positional parameters, so every MaAS test
    phase died with a TypeError AFTER computing its score. Found by the smoke
    gate in 2.2 minutes; it had never surfaced before because a full search
    takes hours and no earlier run survived long enough to reach test().
    DAAO's fork does not carry the mismatch, so this patches only where the
    broken call exists. Costs are passed as zeros: spend is accounted at the
    proxy per namespace, and the authors' cost fields are cosmetic here."""
    # Two call sites carry the same mismatch: optimizer.py's test() and the
    # SEARCH path's evaluation_utils. The second was found by the smoke gate's
    # rerun -- the search swallows it (broad except, "Score for round 1: None")
    # AFTER the controller is trained and saved, so search exits 0 and only the
    # round bookkeeping is lost. Every earlier MaAS search had been doing this
    # silently; the smoke log made it visible.
    sites = (
        (pkg / "ext" / "maas" / "scripts" / "optimizer.py",
         "self.data_utils.create_result_data(self.round, score)",
         "self.data_utils.create_result_data(self.round, score, 0.0, 0.0, 0)",
         "test() result-data call"),
        (pkg / "ext" / "maas" / "scripts" / "optimizer_utils" / "evaluation_utils.py",
         "optimizer.data_utils.create_result_data(cur_round, score)",
         "optimizer.data_utils.create_result_data(cur_round, score, 0.0, 0.0, 0)",
         "search-path result-data call"),
    )
    for path, broken, fixed, label in sites:
        if not path.exists():
            report(False, f"{path.name} missing")
            continue
        text = path.read_text(encoding="utf-8")
        if fixed in text:
            report(True, f"{label} already fixed")
        elif broken not in text:
            report(True, f"{label} not broken in this fork")
        else:
            path.write_text(text.replace(broken, fixed, 1), encoding="utf-8")
            report(fixed in path.read_text(encoding="utf-8"), f"{label} fixed")


def verify_syntax(path: Path) -> None:
    """Parse a file we just wrote. A patch that produces a SyntaxError is worse
    than no patch: it fails at import time, far from the edit that caused it."""
    import ast

    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        report(False, f"{path.name} does not parse after patching: {exc}")


def seed_workspaces(pkg: Path) -> None:
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    if not base.exists():
        report(False, f"missing {base.relative_to(ROOT)}")
        return
    seeded = 0
    for key, source_name in WORKSPACE_SOURCE.items():
        source = base / source_name
        if not source.exists():
            report(False, f"workspace source {source_name} missing")
            continue
        target = base / key
        for split_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            destination = target / split_dir.name
            if destination.exists():
                continue
            # A workspace seed is code, not a snapshot of an author's previous
            # experiment. Some upstream trees (notably DAAO/HumanEval) commit
            # results.json, CSVs, logs, and controller checkpoints beside the
            # graph. Copying the whole split silently mixes those records into a
            # new SHARED_* run. The optimizer only needs graph.py and template/;
            # it creates round directories and checkpoints itself.
            graph = split_dir / "graph.py"
            template = split_dir / "template"
            if not graph.exists() or not template.is_dir():
                report(False, f"workspace source {source_name}/{split_dir.name} "
                       "is missing graph.py or template/")
                continue
            destination.mkdir(parents=True)
            shutil.copy2(graph, destination / "graph.py")
            shutil.copytree(
                template,
                destination / "template",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            seeded += 1
    present = [k for k in WORKSPACE_SOURCE if (base / k).exists()]
    report(len(present) == len(WORKSPACE_SOURCE),
           f"workspaces seeded ({len(present)}/{len(WORKSPACE_SOURCE)}, {seeded} new dirs)")


def link_data(pkg: Path) -> None:
    data_dir = pkg / "ext" / "maas" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Remove only links this installer made; author files are left in place and
    # restored separately with git.
    for name in LEGACY_LINKS:
        stale = data_dir / name
        if stale.is_symlink():
            stale.unlink()

    for key, (eval_file, search_file, _qtype) in DATASETS.items():
        for suffix, source_name in (("test", eval_file), ("train", search_file)):
            source = DATA / source_name
            if not source.exists():
                report(False, f"missing shared split {source_name}")
                continue
            target = data_dir / f"{key.lower()}_{suffix}.jsonl"
            if target.is_symlink() or target.exists():
                target.unlink()
            try:
                target.symlink_to(os.path.relpath(source, start=target.parent))
            except OSError:
                shutil.copyfile(source, target)
    made = sorted(p.name for p in data_dir.glob("*.jsonl"))
    report(len(made) >= 2 * len(DATASETS), f"data wired ({len(made)} files in {data_dir.relative_to(ROOT)})")


def check(pkg: Path, label: str) -> None:
    print(f"\n[{label}] verification")
    shim = pkg / "ext" / "maas" / "benchmark" / "shared_shim.py"
    report(shim.exists(), "shim present")
    evaluator = pkg / "ext" / "maas" / "scripts" / "evaluator.py"
    report(MARKER in evaluator.read_text(encoding="utf-8"), "evaluator patched")
    base_ops = pkg / "ext" / "maas" / "scripts" / "optimized"
    degraded = [p for p in base_ops.glob("*/*/template/operator.py")
                if "ScEnsemble must degrade" in p.read_text(encoding="utf-8")]
    report(not degraded,
           "ScEnsemble not carrying the withdrawn degradation patch"
           + ("" if not degraded else f" (still patched: {len(degraded)} file(s))"))
    # The label-space guard is the opposite case: it must be present, because
    # without it MMLU-Pro discards over half its samples inside the author's own
    # ScEnsemble. Checked by the marker class name so a partial write is visible.
    operators = list(base_ops.glob("*/*/template/operator.py"))
    guarded = [p for p in operators
               if SCENSEMBLE_PATCH_MARKER in p.read_text(encoding="utf-8")]
    # The Test operator calls this helper, so its absence is a runtime failure that
    # no amount of grepping for the class name would have caught.
    missing_helper = [p for p in operators
                      if "_shim_code_dataset" not in p.read_text(encoding="utf-8")]
    report(not missing_helper,
           f"_shim_code_dataset present in every template "
           f"({len(operators) - len(missing_helper)}/{len(operators)})")
    report(operators and len(guarded) == len(operators),
           f"ScEnsemble label-space guard present ({len(guarded)}/{len(operators)} operator files)")
    configs = pkg / "ext" / "maas" / "benchmark" / "experiment_configs.py"
    report(MARKER in configs.read_text(encoding="utf-8"), "experiment_configs patched")
    base = pkg / "ext" / "maas" / "scripts" / "optimized"
    for key in DATASETS:
        report((base / key).exists(), f"workspace {key}")
    # MBPP public tests: assert coverage, not mere existence. The authors' file
    # exists and is 66% short of our split -- that is what silently discarded 225
    # samples, so the check has to measure the overlap.
    public = pkg / "ext" / "maas" / "data" / "shared_mbpp_public_test.jsonl"
    if public.exists():
        identities = {(json.loads(l).get("entry_point"), json.loads(l).get("prompt"))
                      for l in public.read_text(encoding="utf-8").splitlines()
                      if l.strip()}
        missing = 0
        total = 0
        for split in ("mbpp.jsonl", "mbpp_search.jsonl"):
            path = DATA / split
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                if (row.get("entry_point"), row.get("prompt")) not in identities:
                    missing += 1
        report(missing == 0,
               f"MBPP public tests cover every item ({total - missing}/{total}; "
               f"{len(identities)} identities on file)")
    else:
        report(False, "mbpp_public_test.jsonl missing: the Test operator will raise")

    # ...and that the code actually reads it. The file being complete is useless if
    # the operator still asks for HumanEval's.
    # Every template, not just SHARED_MBPP's: the workspaces share one operator
    # module, and the copy under SHARED_MBPP is never imported.
    templates = list((pkg / "ext" / "maas" / "scripts" / "optimized")
                     .glob("*/*/template/operator.py"))
    hardcoded = [p for p in templates
                 if re.search(r'dataset=(?:"|\')(?:HumanEval|MBPP)(?:"|\')',
                              p.read_text(encoding="utf-8"))]
    report(bool(templates) and not hardcoded,
           f"Test operator benchmark chosen at runtime "
           f"({len(templates) - len(hardcoded)}/{len(templates)} template(s))"
           + ("" if not hardcoded else f"; still hardcoded: {[p.parts[-4] for p in hardcoded]}"))
    utils_text = (pkg / "ext" / "maas" / "scripts" / "utils.py").read_text(
        encoding="utf-8")
    code_templates = [p for p in templates
                      if "class Test" in p.read_text(encoding="utf-8")]
    report('dataset == "SHARED_MBPP"' in utils_text and "len(task_matches) == 1" in utils_text,
           "MBPP public tests are disambiguated by task text")
    report(bool(code_templates) and all("problem=problem" in
                                       p.read_text(encoding="utf-8")
                                       for p in code_templates),
           "MBPP Test operators pass task identity")

    data_dir = pkg / "ext" / "maas" / "data"
    for key in DATASETS:
        for suffix in ("train", "test"):
            path = data_dir / f"{key.lower()}_{suffix}.jsonl"
            portable = not path.is_symlink() or not Path(os.readlink(path)).is_absolute()
            ok = path.exists() and path.stat().st_size > 0 and portable
            report(ok, f"data {path.name}")

    operator_files = list((pkg / "ext" / "maas" / "scripts" / "optimized")
                         .glob("*/*/template/operator.py"))
    exec_problems = [p for f in operator_files for p in exec_guard_patch.verify_file(f)]
    report(not exec_problems,
           f"generated-code execution killable in every template "
           f"({len(operator_files)} file(s))"
           + ("" if not exec_problems else f"; {exec_problems[:3]}"))

    ns_problems = namespace_patch.verify_file(
        pkg / "provider" / "llm_provider_registry.py")
    report(not ns_problems, "namespace override present"
           + ("" if not ns_problems else f" ({ns_problems})"))

    provider_init = pkg / "provider" / "__init__.py"
    report(provider_init.exists()
           and patch_providers.MARKER in provider_init.read_text(encoding="utf-8"),
           "unused provider SDK imports are optional")

    check_prompt_wording(pkg)

    node = pkg / "actions" / "action_node.py"
    report(node.exists() and "_shim_dedent_code(content)" in node.read_text(encoding="utf-8"),
           "code_fill dedents model replies before sanitize")
    sanitizer = pkg / "utils" / "sanitize.py"
    report(sanitizer.exists() and "_shim_dedent_code(code)" in sanitizer.read_text(encoding="utf-8"),
           "sanitize dedents model replies for every caller")

    if pkg.parent.name == "daao":
        link = pkg.parent / "all-MiniLM-L6-v2"
        report(link.is_symlink()
               and os.readlink(link) == "../../shared/models/all-MiniLM-L6-v2"
               and link.resolve().is_dir(),
               "DAAO MiniLM path is a portable relative link")


def _prompt_namespace(path: Path, dataset: str) -> dict:
    """Execute a prompt module as the given dataset and return its constants.

    Executed rather than parsed, because both the escape repair and the dataset
    override happen at runtime: `ast` sees the author's original text and would
    report success no matter what the appended block does. SHIM_DATASET is set for
    the duration, the way sweep.py sets it per job.
    """
    previous = os.environ.get("SHIM_DATASET")
    os.environ["SHIM_DATASET"] = dataset
    try:
        namespace: dict = {"__name__": "shim_prompt_probe"}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        return namespace
    finally:
        if previous is None:
            os.environ.pop("SHIM_DATASET", None)
        else:
            os.environ["SHIM_DATASET"] = previous


def check_prompt_wording(pkg: Path) -> None:
    """Assert the repair took, the adaptation took, and MATH/AMC did not move.

    The last of the three is the one worth checking hardest: an adaptation that
    also changes the maths wording would silently make every MATH and AMC number
    incomparable with the author's published protocol.
    """
    live = pkg / "ext" / "maas" / "scripts" / "optimized" / "MATH" / "train" / "template"
    if not (live / "prompt.py").exists():
        report(False, "cannot verify prompt wording: MATH/train/template missing")
        return

    control = {"\x07", "\x08", "\x0b", "\x0c"}
    leftovers: list[str] = []
    for dataset in ("SHARED_MATH", "SHARED_DROP", "SHARED_MMLUPRO"):
        for name in ("prompt.py", "op_prompt.py"):
            for key, value in _prompt_namespace(live / name, dataset).items():
                if key.endswith("_PROMPT") and isinstance(value, str):
                    if control & set(value):
                        leftovers.append(f"{dataset}/{name}:{key}")
    report(not leftovers,
           "no mis-escaped LaTeX left in any prompt constant"
           + ("" if not leftovers else f" (still corrupted: {leftovers[:4]})"))

    maths = _prompt_namespace(live / "prompt.py", "SHARED_MATH")
    drop = _prompt_namespace(live / "prompt.py", "SHARED_DROP")
    mmlu = _prompt_namespace(live / "prompt.py", "SHARED_MMLUPRO")

    # The author's maths wording survives, with \boxed now readable.
    report("mathematical problem" in maths["GENERATE_SOLUTION_PROMPT"]
           and "\\boxed{}" in maths["GENERATE_SOLUTION_PROMPT"],
           "MATH keeps the author's wording, with \\boxed repaired")

    # DROP and MMLU-Pro no longer claim to be maths, and ask for the format the
    # grader actually reads.
    report("mathematical problem" not in drop["GENERATE_SOLUTION_PROMPT"]
           and "Answer: <answer>" in drop["GENERATE_SOLUTION_PROMPT"]
           and "\\boxed" not in drop["GENERATE_SOLUTION_PROMPT"],
           "DROP asks for a passage answer, not a boxed maths answer")
    report("mathematical problem" not in mmlu["GENERATE_SOLUTION_PROMPT"]
           and "Answer: (X)" in mmlu["GENERATE_SOLUTION_PROMPT"]
           and "\\boxed" not in mmlu["GENERATE_SOLUTION_PROMPT"],
           "MMLU-Pro asks for an option letter, not a boxed maths answer")

    # Strategy prompts are the method under test and must be byte-identical across
    # datasets. SC_ENSEMBLE_PROMPT in particular: its letter-space fix is a
    # separate, deliberate change and must not be entangled with wording.
    ops = {d: _prompt_namespace(live / "op_prompt.py", d)
           for d in ("SHARED_MATH", "SHARED_DROP", "SHARED_MMLUPRO")}
    for key in ("SC_ENSEMBLE_PROMPT", "SELFREFINE_PROMPT"):
        values = {d: ns.get(key) for d, ns in ops.items()}
        report(len(set(values.values())) == 1,
               f"{key} identical on every dataset (author's design untouched)")

    # The Programmer operator still writes code on all three; only what the code is
    # about changes. A dropped `solve` contract would break the operator silently.
    for dataset, ns in ops.items():
        verifier = ns.get("PYTHON_CODE_VERIFIER_PROMPT", "")
        report("Define a function named `solve`" in verifier
               and "{problem}" in verifier and "{analysis}" in verifier,
               f"Programmer contract and placeholders intact on {dataset}")

    # The op_prompt.py constants, asserted on CONTENT and not only on placeholders.
    #
    # The earlier version of this check only asked whether "{instruction}" and
    # "{input}" survived -- which the author's unmodified prompt also satisfies, so
    # it passed while the adaptation was silently not running at all. What has to be
    # asserted is that the maths worked examples are gone and the new ones are there.
    expectations = {
        "SHARED_DROP": ("Reading Comprehension Reasoning Instruction",
                        "reading comprehension question and its passage"),
        "SHARED_MMLUPRO": ("Multiple-Choice Reasoning Instruction",
                           "multiple-choice question and output the answer"),
    }
    for dataset, (cot_marker, verifier_marker) in expectations.items():
        namespace = _prompt_namespace(live / "op_prompt.py", dataset)
        cot = namespace.get("GENERATE_COT_PROMPT", "")
        verifier = namespace.get("PYTHON_CODE_VERIFIER_PROMPT", "")
        report("{instruction}" in cot and "{input}" in cot,
               f"GENERATE_COT_PROMPT keeps its placeholders on {dataset}")
        report(cot_marker in cot and "Mathematical Reasoning Instruction" not in cot
               and "Cone with height 12cm" not in cot and "\\boxed" not in cot,
               f"GENERATE_COT_PROMPT is really adapted on {dataset} "
               f"(no GSM8K/MATH worked examples left)")
        report(verifier_marker in verifier
               and "based on a given mathematical problem" not in verifier,
               f"PYTHON_CODE_VERIFIER_PROMPT is really adapted on {dataset}")
    # And the maths datasets must still carry the author's examples.
    maths_cot = _prompt_namespace(live / "op_prompt.py", "SHARED_MATH").get(
        "GENERATE_COT_PROMPT", "")
    report("Mathematical Reasoning Instruction" in maths_cot
           and "Cone with height 12cm" in maths_cot,
           "MATH keeps the author's worked examples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = parser.parse_args()

    for label, pkg in REPOS.items():
        if not pkg.exists():
            report(False, f"{label}: package dir missing at {pkg}")
            continue
        if args.check:
            check(pkg, label)
            continue
        print(f"\n[{label}] installing")
        patch_providers.patch(label, pkg / "provider" / "__init__.py")
        install_shim(pkg)
        patch_evaluator(pkg)
        patch_experiment_configs(pkg)
        seed_workspaces(pkg)
        patch_prompt_formatting(pkg)
        patch_prompt_task_wording(pkg)
        patch_code_execution(pkg)
        patch_indented_code(pkg)
        link_sentence_model(pkg)
        patch_namespace_override(pkg)
        patch_test_result_data(pkg)
        # After patch_prompt_formatting, not before: that function skips any file
        # already containing "_shim_safe_format", and the block appended here
        # mentions it -- running in the other order would silently skip the
        # brace-safe rewrite on a fresh checkout.
        patch_scensemble_labels(pkg)
        # After link_data would be wrong: link_data symlinks the shared splits into
        # the same directory, and this writes a real file there.
        write_mbpp_public_tests(pkg)
        patch_mbpp_test_dataset(pkg)
        patch_mbpp_test_identity(pkg)
        link_data(pkg)
        write_config(pkg.parent, label)

    problems.extend(patch_providers.problems)

    print("\n" + "=" * 60)
    if problems:
        print(f"MaAS-family shim: {len(problems)} problem(s)")
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print("MaAS-family shim OK")


if __name__ == "__main__":
    main()
