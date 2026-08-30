#!/usr/bin/env python3
"""Verify the MaAS-family shim grades and returns the arity each repo expects.

Run from the repo being tested so its package resolves, e.g.

    cd third_party/maas && python ../../shims/maas_family/test_shim.py maas
    cd third_party/daao && python ../../shims/maas_family/test_shim.py daao

A fake graph returns a known-correct answer, so a working shim must score 1.0.
The arity matters as much as the score: DAAO needs the seventh `vae` element and
its `is_solved` flag, and a shim that silently dropped it would still look like a
successful run while the difficulty estimator learned nothing.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

package = sys.argv[1] if len(sys.argv) > 1 else "maas"
expects_vae = package == "daao"

sys.path.insert(0, str(Path.cwd()))

shim = importlib.import_module(f"{package}.ext.maas.benchmark.shared_shim")
shared_bench = shim.shared_bench

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


print(f"=== {package}: shim import ===")
check("EXPECTS_VAE matches repo", shim.EXPECTS_VAE == expects_vae,
      f"EXPECTS_VAE={shim.EXPECTS_VAE}, expected {expects_vae}")
check("5 datasets registered", len(shim.SHARED_DATASET_CONFIGS) == 5,
      str(sorted(shim.SHARED_DATASET_CONFIGS)))


def gold_answer(name: str, row: dict) -> str:
    if name in ("math", "amc"):
        return f"The answer is \\boxed{{{row['answer']}}}"
    if name == "mbpp":
        return row["code"]
    if name == "drop":
        return f"Answer: {row['answers'][0]}"
    if name == "mmlu_pro":
        return f"Answer: ({row['answer']})"
    raise KeyError(name)


class FakeGraph:
    """Stands in for a searched workflow, returning the repo's expected tuple."""

    def __init__(self, reply: str, with_vae: bool):
        self.reply = reply
        self.with_vae = with_vae

    async def __call__(self, prompt, *extra):
        import torch

        logprob = torch.tensor(0.0)
        if self.with_vae:
            vae = {"z_difficulty": torch.zeros((1, 32)), "difficulty_scalar": torch.tensor(0.5),
                   "mu": torch.zeros((1, 32)), "logvar": torch.zeros((1, 32))}
            return self.reply, 0.0, logprob, vae
        return self.reply, 0.0, logprob


async def run() -> None:
    for key, cls in sorted(shim.SHARED_DATASET_CONFIGS.items()):
        dataset = cls.SHARED_DATASET
        row = shared_bench.load(dataset)[0]
        # BaseBenchmark.__init__ moves the controller onto a device, so it needs a
        # real module even though grading never touches it.
        import torch

        dummy_controller = torch.nn.Linear(1, 1)
        benchmark = cls(name=key, file_path="unused", log_path="/tmp",
                        batch_size=1, controller=dummy_controller,
                        operator_embeddings=torch.zeros((1, 4)),
                        optimizer=torch.optim.SGD(dummy_controller.parameters(), lr=0.0))
        graph = FakeGraph(gold_answer(dataset, row), expects_vae)
        result = await benchmark.evaluate_problem(dict(row), graph)

        expected_len = 7 if expects_vae else 6
        check(f"{key}: tuple arity {expected_len}", len(result) == expected_len, f"got {len(result)}")
        score = result[3]
        check(f"{key}: gold answer scores 1.0", score >= 0.99, f"score={score}")
        if expects_vae:
            vae = result[6]
            check(f"{key}: vae['is_solved'] set", vae.get("is_solved") == 1, str(vae.get("is_solved")))

        wrong = FakeGraph("completely wrong -12345", expects_vae)
        bad = await benchmark.evaluate_problem(dict(row), wrong)
        check(f"{key}: wrong answer scores 0", bad[3] < 0.5, f"score={bad[3]}")


asyncio.run(run())

print("\n" + "=" * 56)
if failures:
    print(f"{package} SHIM TEST FAILED ({len(failures)})")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print(f"{package} SHIM TEST PASSED")
