#!/usr/bin/env python3
"""Fail when a runtime prompt names an unrelated benchmark.

The MaAS-family prompt repair is selected at module execution time.  Scanning the
source text therefore reports comments, inactive branches, and the author's
overridden prompt as false positives.  This audit executes each prompt module in
the same dataset environment as the runner and inspects only live prompt strings.
"""

from __future__ import annotations

import os
import importlib
import inspect
import json
import re
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    # The authors' MATH prompt explicitly labels its worked examples
    # "GSM8K/MATH style".  AMC intentionally reuses that maths prompt.
    "SHARED_MATH": {"MATH", "GSM8K"},
    "SHARED_AMC": {"MATH", "AMC", "GSM8K"},
    "SHARED_MBPP": {"MBPP"},
    "SHARED_DROP": {"DROP"},
    "SHARED_MMLUPRO": {"MMLU", "MMLU-Pro", "MMLU_Pro"},
}
FOREIGN = (
    "HumanEval", "GSM8K", "MBPP", "MATH", "DROP", "MMLU",
    "HotpotQA", "LiveCodeBench", "Codeforces",
)
SEARCH_ROOTS = (
    "third_party/maas/maas/ext/maas/scripts/optimized",
    "third_party/daao/daao/ext/maas/scripts/optimized",
    "third_party/aflow/workspace",
    "third_party/flowbank/DiverseFlow/workspace",
)
PROMPT_FILES = {"prompt.py", "op_prompt.py", "prompt_custom.py"}

# Dataset names catch direct copy/paste contamination. These phrases catch the
# more dangerous variant where a prompt describes the wrong answer space without
# naming the benchmark it came from.
WRONG_TASK_PHRASES = {
    "SHARED_MBPP": (r"\\boxed",),
    "SHARED_DROP": (
        r"\\boxed", r"offered 4", r"4 (?:answers|letters)",
        r"A, B, C (?:and|or) D", r"complex math problem",
        r"shortest exact span", r"copied rather than paraphrased",
    ),
    "SHARED_MMLUPRO": (r"offered 4", r"4 (?:answers|letters)",
                        r"A, B, C (?:and|or) D", r"complex math problem"),
}


def runtime_prompts(path: Path, dataset: str) -> dict[str, str]:
    """Return the prompt constants that survive execution for one dataset."""
    previous = os.environ.get("SHIM_DATASET")
    os.environ["SHIM_DATASET"] = dataset
    try:
        namespace: dict = {"__name__": "prompt_contamination_probe"}
        source = path.read_text(encoding="utf-8", errors="strict")
        exec(compile(source, str(path), "exec"), namespace)
        return {
            name: value
            for name, value in namespace.items()
            if name.endswith("_PROMPT") and isinstance(value, str)
        }
    finally:
        if previous is None:
            os.environ.pop("SHIM_DATASET", None)
        else:
            os.environ["SHIM_DATASET"] = previous


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_strings(item)


def install_registry_stub() -> None:
    """Provide the tiny dependency prompt modules need, without loading ML code."""
    if "class_registry" in sys.modules:
        return

    module = types.ModuleType("class_registry")

    class ClassRegistry:
        def __init__(self):
            self._classes = {}

        def register(self, name, *args, **kwargs):
            def decorator(cls):
                self._classes[name] = cls
                return cls
            return decorator

        def keys(self):
            return self._classes.keys()

        def get(self, name, *args, **kwargs):
            return self._classes[name](*args, **kwargs)

        def get_class(self, name):
            return self._classes[name]

    module.ClassRegistry = ClassRegistry
    sys.modules["class_registry"] = module


def family_prompt_values(repo: Path, package: str, dataset: str):
    """Render every callable prompt-set field for G-Designer or CARD."""
    install_registry_stub()
    sys.path.insert(0, str(repo))
    try:
        importlib.import_module(f"{package}.prompt.shared_prompt_sets")
        registry_module = importlib.import_module(
            f"{package}.prompt.prompt_set_registry")
        prompt_set = registry_module.PromptSetRegistry.get(dataset)

        samples = {
            "question": "TASK_TEXT", "query": "QUERY_TEXT", "file": "FILE_TEXT",
            "results": "RESULT_TEXT", "answer": "ANSWER_TEXT",
            "solution": "SOLUTION_TEXT", "feedback": "FEEDBACK_TEXT",
            "constraint": "CONSTRAINT_TEXT", "materials": {},
            "answers": ["ANSWER_ONE", "ANSWER_TWO"],
        }
        # These are the prompt-set methods reached by the agent constructors and
        # forward paths. Utility prompts for disabled web/file tools are excluded.
        live_methods = {
            "get_constraint", "get_description", "get_format",
            "get_answer_prompt", "get_analyze_constraint",
            "get_adversarial_answer_prompt", "get_decision_constraint",
            "get_decision_role", "get_decision_few_shot",
        }
        for method_name in sorted(live_methods):
            if not hasattr(prompt_set, method_name):
                continue
            method = getattr(prompt_set, method_name)
            if not callable(method):
                continue
            signature = inspect.signature(method)
            parameters = [p for p in signature.parameters.values()
                          if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
            owner = sys.modules.get(getattr(method, "__module__", ""))
            descriptions = getattr(owner, "ROLE_DESCRIPTION", {})
            roles = sorted(descriptions) if isinstance(descriptions, dict) else []
            role_values = roles if any(p.name == "role" for p in parameters) else [None]
            if any(p.name == "role" for p in parameters) and not role_values:
                raise RuntimeError(f"{package}.{dataset}.{method_name} has no role source")
            for role in role_values:
                kwargs = {}
                unknown = []
                for parameter in parameters:
                    if parameter.name == "role":
                        kwargs[parameter.name] = role
                    elif parameter.name in samples:
                        kwargs[parameter.name] = samples[parameter.name]
                    elif parameter.default is inspect.Parameter.empty:
                        unknown.append(parameter.name)
                if unknown:
                    raise RuntimeError(
                        f"cannot render {package}.{dataset}.{method_name}; "
                        f"unknown arguments {unknown}")
                value = method(**kwargs)
                for index, text in enumerate(iter_strings(value)):
                    yield f"{package}.{method_name}[{role or index}]", text
    finally:
        sys.path.pop(0)


def masrouter_prompt_values(dataset: str):
    base = ROOT / "third_party" / "masrouter" / "MAR" / "Roles"
    routing = {
        "math": ("Math", "math.json"),
        "amc": ("Math", "math.json"),
        "mbpp": ("Code", "mbpp.json"),
        "drop": ("Commonsense_drop", "shared_drop.json"),
        "mmlu_pro": ("Commonsense_mmlu_pro", "shared_mmlu_pro.json"),
    }
    role_dir, final_name = routing[dataset]
    paths = sorted((base / role_dir).glob("*.json"))
    paths.append(base / "FinalNode" / final_name)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for index, value in enumerate(iter_strings(payload)):
            yield f"masrouter/{path.relative_to(base)}[{index}]", value


def dataset_name(key: str) -> str:
    return {
        "SHARED_MATH": "math",
        "SHARED_AMC": "amc",
        "SHARED_MBPP": "mbpp",
        "SHARED_DROP": "drop",
        "SHARED_MMLUPRO": "mmlu_pro",
    }[key]


def inspect_prompt(findings, source, dataset, variable, value) -> None:
    for name in FOREIGN:
        if name in EXPECTED[dataset]:
            continue
        # Benchmark names are proper names. Case-sensitive matching deliberately
        # does not mistake a domain-expert role's ordinary phrase "math games"
        # for a claim that the current benchmark is MATH.
        match = re.search(rf"\b{re.escape(name)}\b", value)
        if match:
            context = " ".join(
                value[max(0, match.start() - 60):match.end() + 60].split())
            findings.append((source, dataset, variable, name, context))
    for pattern in WRONG_TASK_PHRASES.get(dataset, ()):
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            context = " ".join(
                value[max(0, match.start() - 60):match.end() + 60].split())
            findings.append((source, dataset, variable, pattern, context))


def main() -> None:
    findings = []
    failures = []
    scanned_files = 0
    scanned_prompts = 0
    sys.path.insert(0, str(ROOT / "shared"))
    import bench as shared_bench
    for dataset, value in shared_bench.ANSWER_FORMAT.items():
        key = "SHARED_" + dataset.upper().replace("_", "")
        inspect_prompt(findings, Path("shared/bench.py"), key,
                       f"ANSWER_FORMAT[{dataset}]", value)
        scanned_prompts += 1
    for relative in SEARCH_ROOTS:
        base = ROOT / relative
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            dataset = next(
                (part for part in path.parts if part.startswith("SHARED_")), None
            )
            if dataset not in EXPECTED or path.name not in PROMPT_FILES:
                continue
            scanned_files += 1
            try:
                prompts = runtime_prompts(path, dataset)
            except Exception as exc:  # A broken prompt module is itself a failure.
                failures.append((path.relative_to(ROOT), exc))
                continue
            scanned_prompts += len(prompts)
            for variable, value in prompts.items():
                inspect_prompt(findings, path.relative_to(ROOT), dataset, variable, value)

    for repo_name, package in (("gdesigner", "GDesigner"), ("card", "CARD")):
        repo = ROOT / "third_party" / repo_name
        if not repo.exists():
            continue
        for key in EXPECTED:
            dataset = dataset_name(key)
            try:
                values = list(family_prompt_values(repo, package, dataset))
            except Exception as exc:
                failures.append((Path(f"third_party/{repo_name}/{dataset}"), exc))
                continue
            scanned_files += 1
            scanned_prompts += len(values)
            for variable, value in values:
                inspect_prompt(findings, Path(f"third_party/{repo_name}"),
                               key, variable, value)

    masrouter = ROOT / "third_party" / "masrouter"
    if masrouter.exists():
        for key in EXPECTED:
            dataset = dataset_name(key)
            try:
                values = list(masrouter_prompt_values(dataset))
            except Exception as exc:
                failures.append((Path(f"third_party/masrouter/{dataset}"), exc))
                continue
            scanned_files += 1
            scanned_prompts += len(values)
            for variable, value in values:
                inspect_prompt(findings, Path("third_party/masrouter"),
                               key, variable, value)

    for path, exc in failures:
        print(f"[FAIL] {path}: cannot evaluate prompt module: {exc}")
    for path, dataset, variable, name, context in findings:
        print(
            f"[FAIL] {path}: {dataset}/{variable} mentions {name!r}\n"
            f"  ...{context}..."
        )
    if failures or findings:
        raise SystemExit(
            f"{len(failures)} module error(s), "
            f"{len(findings)} cross-dataset prompt mention(s)"
        )
    print(
        "prompt contamination scan OK "
        f"({scanned_prompts} live prompts in {scanned_files} files)"
    )


if __name__ == "__main__":
    main()
