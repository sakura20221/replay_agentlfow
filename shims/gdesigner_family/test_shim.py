#!/usr/bin/env python3
"""Verify the G-Designer-family shim before any GPU time is spent.

Run from inside the repo under test:

    cd third_party/gdesigner && python ../../shims/gdesigner_family/test_shim.py GDesigner
    cd third_party/card      && python ../../shims/gdesigner_family/test_shim.py CARD

Checks the three things that would otherwise fail silently: the five prompt
domains actually register (an unregistered domain makes Graph fall back or
raise), the item shape matches what the authors' training loop reads, and the
shared scorer grades in both directions. The derived runner is compiled rather
than executed, since executing it needs the whole graph and an LLM.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

package = sys.argv[1] if len(sys.argv) > 1 else "GDesigner"
sys.path.insert(0, str(Path.cwd()))

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


print(f"=== {package}: prompt domains ===")
prompt_module = importlib.import_module(f"{package}.prompt.shared_prompt_sets")
registry = importlib.import_module(f"{package}.prompt.prompt_set_registry").PromptSetRegistry

for domain in prompt_module.SHARED_DOMAINS:
    try:
        instance = registry.get(domain)
        check(f"domain {domain!r} registered", instance is not None, type(instance).__name__)
    except Exception as exc:  # noqa: BLE001
        check(f"domain {domain!r} registered", False, f"{type(exc).__name__}: {exc}")

print(f"\n=== {package}: shared data + scoring ===")
shared_dataset = importlib.import_module("datasets.shared_dataset")
shared_bench = shared_dataset.shared_bench

WRONG = {
    "math": "the answer is -999999",
    "amc": "the answer is -999999",
    "mbpp": "def nope():\n    return None\n",
    "drop": "Answer: zzzz nonexistent",
    "mmlu_pro": "I decline to answer.",
}


def gold_reply(name: str, row: dict) -> str:
    if name in ("math", "amc"):
        return f"\\boxed{{{row['answer']}}}"
    if name == "mbpp":
        return row["code"]
    if name == "drop":
        return f"Answer: {row['answers'][0]}"
    if name == "mmlu_pro":
        return f"Answer: ({row['answer']})"
    raise KeyError(name)


for name in shared_bench.DATASETS:
    rows = list(shared_bench.load(name))
    items = shared_dataset.shared_data_process(rows[:4], name)
    shape_ok = all({"task", "step", "answer", "row"} <= set(item) for item in items)
    check(f"{name}: item shape matches the training loop", shape_ok, str(sorted(items[0]))[:70])
    check(f"{name}: task text non-empty", bool(items[0]["task"].strip()), f"{len(items[0]['task'])} chars")

    good = [shared_dataset.shared_score(name, item, gold_reply(name, item["row"]))[0] for item in items]
    check(f"{name}: gold answer scores 1.0", all(v >= 0.99 for v in good),
          str([round(v, 3) for v in good]))
    bad = [shared_dataset.shared_score(name, item, WRONG[name])[0] for item in items]
    check(f"{name}: wrong answer scores 0", all(v < 0.5 for v in bad), str([round(v, 3) for v in bad]))

print(f"\n=== {package}: derived runner ===")
runner = Path("experiments") / "run_shared.py"
check("run_shared.py exists", runner.exists())
if runner.exists():
    try:
        py_compile.compile(str(runner), doraise=True)
        check("run_shared.py compiles", True)
    except py_compile.PyCompileError as exc:
        check("run_shared.py compiles", False, str(exc)[:200])
    text = runner.read_text(encoding="utf-8")
    check("no leftover gsm_get_predict", "gsm_get_predict" not in text)
    check("scores via shared_score", "shared_score(args.domain" in text)
    check("domain follows the flag", 'domain="gsm8k"' not in text)

print("\n" + "=" * 58)
if failures:
    print(f"{package} SHIM TEST FAILED ({len(failures)})")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print(f"{package} SHIM TEST PASSED")
